from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from queue_proxy.task_queue import (
    ACTIVE_STATES,
    CLAIMABLE_STATES,
    DEFAULT_MAX_OUTPUT_TOKENS,
    QueueTask,
    StoredTask,
    TaskStore,
    context_plan_for_model,
    now_iso,
)

TASK_COLUMNS = (
    "task_id",
    "tenant",
    "project",
    "service",
    "task",
    "job_id",
    "idempotency_key",
    "priority",
    "model",
    "endpoint",
    "estimated_input_tokens",
    "max_output_tokens",
    "required_context_tokens",
    "state",
    "payload_json",
    "orchestration_json",
    "artifacts_json",
    "labels_json",
    "created_at",
    "updated_at",
    "attempt_count",
    "started_at",
    "finished_at",
    "next_attempt_at",
    "result_json",
    "error_json",
)
TASK_COLUMNS_SQL = ", ".join(TASK_COLUMNS)
TASK_RETURNING_SQL = f"RETURNING {TASK_COLUMNS_SQL}"


class PostgresTaskStore(TaskStore):
    def __init__(
        self,
        dsn: str,
        *,
        connect_factory: Callable[[], Any] | None = None,
        ensure_schema: bool = True,
    ) -> None:
        if not dsn:
            raise ValueError("PostgresTaskStore requires a DSN.")
        self.dsn = dsn
        self.connect_factory = connect_factory
        if ensure_schema:
            self.ensure_schema()

    def submit_many(self, tasks: list[QueueTask]) -> tuple[list[StoredTask], list[StoredTask]]:
        accepted: list[StoredTask] = []
        reused: list[StoredTask] = []
        with self.connection() as conn:
            with conn.cursor() as cur:
                for task in tasks:
                    stored = self.insert_task(cur, task)
                    if stored is None:
                        cur.execute(
                            f"""
                            SELECT {TASK_COLUMNS_SQL}
                            FROM llmo_tasks
                            WHERE tenant = %s AND idempotency_key = %s
                            """,
                            (task.tenant, task.idempotency_key),
                        )
                        row = cur.fetchone()
                        if row is not None:
                            reused.append(stored_task_from_row(row))
                        continue
                    accepted.append(stored)
            conn.commit()
        return accepted, reused

    def queue_lengths_by_model(self) -> dict[str, int]:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT model, COUNT(*) AS count
                    FROM llmo_tasks
                    WHERE state = ANY(%s)
                    GROUP BY model
                    """,
                    (list(ACTIVE_STATES),),
                )
                rows = cur.fetchall()
        return {str(row["model"]): int(row["count"]) for row in rows}

    def context_plans_by_model(self) -> dict[str, dict[str, Any]]:
        tasks_by_model: dict[str, list[StoredTask]] = {}
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT {TASK_COLUMNS_SQL}
                    FROM llmo_tasks
                    WHERE state = ANY(%s)
                    ORDER BY model, created_at, task_id
                    """,
                    (list(ACTIVE_STATES),),
                )
                rows = cur.fetchall()
        for row in rows:
            task = stored_task_from_row(row)
            tasks_by_model.setdefault(task.model, []).append(task)
        return {
            model: context_plan_for_model(tasks)
            for model, tasks in sorted(tasks_by_model.items())
        }

    def get_task(self, tenant: str, task_id: str) -> StoredTask | None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT {TASK_COLUMNS_SQL}
                    FROM llmo_tasks
                    WHERE tenant = %s AND task_id = %s
                    """,
                    (tenant, task_id),
                )
                row = cur.fetchone()
        return stored_task_from_row(row) if row is not None else None

    def list_tasks(
        self,
        tenant: str,
        *,
        state: str | None = None,
        model: str | None = None,
        limit: int = 100,
    ) -> list[StoredTask]:
        filters = ["tenant = %s"]
        params: list[Any] = [tenant]
        if state is not None:
            filters.append("state = %s")
            params.append(state)
        if model is not None:
            filters.append("model = %s")
            params.append(model)
        params.append(max(1, limit))
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT {TASK_COLUMNS_SQL}
                    FROM llmo_tasks
                    WHERE {" AND ".join(filters)}
                    ORDER BY created_at, task_id
                    LIMIT %s
                    """,
                    tuple(params),
                )
                rows = cur.fetchall()
        return [stored_task_from_row(row) for row in rows]

    def claim_next(self, *, model: str | None = None) -> StoredTask | None:
        now = now_iso()
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    WITH group_scores AS (
                        SELECT
                            candidate.tenant,
                            candidate.project,
                            candidate.service,
                            candidate.task,
                            candidate.priority,
                            candidate.model,
                            MIN(candidate.created_at) AS oldest_runnable_at,
                            (
                                SELECT MAX(served.updated_at)
                                FROM llmo_tasks served
                                WHERE served.tenant = candidate.tenant
                                  AND served.project = candidate.project
                                  AND served.service = candidate.service
                                  AND served.task = candidate.task
                                  AND served.priority = candidate.priority
                                  AND served.model = candidate.model
                                  AND served.attempt_count > 0
                            ) AS last_claimed_at
                        FROM llmo_tasks candidate
                        WHERE candidate.state = ANY(%s)
                          AND (%s IS NULL OR candidate.model = %s)
                          AND (
                              candidate.next_attempt_at IS NULL
                              OR candidate.next_attempt_at <= %s::timestamptz
                          )
                        GROUP BY
                            candidate.tenant,
                            candidate.project,
                            candidate.service,
                            candidate.task,
                            candidate.priority,
                            candidate.model
                    ),
                    selected_group AS (
                        SELECT *
                        FROM group_scores
                        ORDER BY
                            (last_claimed_at IS NOT NULL),
                            last_claimed_at NULLS FIRST,
                            oldest_runnable_at,
                            tenant,
                            project,
                            service,
                            task,
                            priority,
                            model
                        LIMIT 1
                    ),
                    next_task AS (
                        SELECT candidate.task_id
                        FROM llmo_tasks candidate
                        JOIN selected_group
                          ON selected_group.tenant = candidate.tenant
                         AND selected_group.project = candidate.project
                         AND selected_group.service = candidate.service
                         AND selected_group.task = candidate.task
                         AND selected_group.priority = candidate.priority
                         AND selected_group.model = candidate.model
                        WHERE candidate.state = ANY(%s)
                          AND (%s IS NULL OR candidate.model = %s)
                          AND (
                              candidate.next_attempt_at IS NULL
                              OR candidate.next_attempt_at <= %s::timestamptz
                          )
                        ORDER BY candidate.created_at, candidate.task_id
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                    )
                    UPDATE llmo_tasks
                    SET state = 'running',
                        started_at = COALESCE(started_at, %s),
                        attempt_count = attempt_count + 1,
                        next_attempt_at = NULL,
                        updated_at = %s
                    WHERE task_id IN (SELECT task_id FROM next_task)
                    {TASK_RETURNING_SQL}
                    """,
                    (
                        list(CLAIMABLE_STATES),
                        model,
                        model,
                        now,
                        list(CLAIMABLE_STATES),
                        model,
                        model,
                        now,
                        now,
                        now,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        return stored_task_from_row(row) if row is not None else None

    def record_result(self, task_id: str, result: dict[str, Any]) -> StoredTask:
        now = now_iso()
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE llmo_tasks
                    SET state = 'succeeded',
                        result_json = %s::jsonb,
                        error_json = NULL,
                        next_attempt_at = NULL,
                        finished_at = %s,
                        updated_at = %s
                    WHERE task_id = %s
                    {TASK_RETURNING_SQL}
                    """,
                    (json.dumps(result), now, now, task_id),
                )
                row = cur.fetchone()
            conn.commit()
        if row is None:
            raise KeyError(task_id)
        return stored_task_from_row(row)

    def record_error(self, task_id: str, error: dict[str, Any]) -> StoredTask:
        now = now_iso()
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE llmo_tasks
                    SET state = 'failed',
                        error_json = %s::jsonb,
                        next_attempt_at = NULL,
                        finished_at = %s,
                        updated_at = %s
                    WHERE task_id = %s
                    {TASK_RETURNING_SQL}
                    """,
                    (json.dumps(error), now, now, task_id),
                )
                row = cur.fetchone()
            conn.commit()
        if row is None:
            raise KeyError(task_id)
        return stored_task_from_row(row)

    def record_retry(
        self,
        task_id: str,
        error: dict[str, Any],
        next_attempt_at: str,
    ) -> StoredTask:
        now = now_iso()
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE llmo_tasks
                    SET state = 'queued',
                        error_json = %s::jsonb,
                        next_attempt_at = %s::timestamptz,
                        updated_at = %s
                    WHERE task_id = %s
                    {TASK_RETURNING_SQL}
                    """,
                    (json.dumps(error), next_attempt_at, now, task_id),
                )
                row = cur.fetchone()
            conn.commit()
        if row is None:
            raise KeyError(task_id)
        return stored_task_from_row(row)

    def cancel_task(self, tenant: str, task_id: str) -> StoredTask | None:
        now = now_iso()
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE llmo_tasks
                    SET state = 'cancelled',
                        finished_at = COALESCE(finished_at, %s),
                        next_attempt_at = NULL,
                        updated_at = %s
                    WHERE tenant = %s
                      AND task_id = %s
                      AND state NOT IN ('succeeded', 'failed', 'cancelled')
                    {TASK_RETURNING_SQL}
                    """,
                    (now, now, tenant, task_id),
                )
                row = cur.fetchone()
                if row is None:
                    cur.execute(
                        f"""
                        SELECT {TASK_COLUMNS_SQL}
                        FROM llmo_tasks
                        WHERE tenant = %s AND task_id = %s
                        """,
                        (tenant, task_id),
                    )
                    row = cur.fetchone()
            conn.commit()
        return stored_task_from_row(row) if row is not None else None

    def ensure_schema(self) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS llmo_tasks (
                        task_id TEXT PRIMARY KEY,
                        tenant TEXT NOT NULL,
                        project TEXT NOT NULL,
                        service TEXT NOT NULL,
                        task TEXT NOT NULL,
                        job_id TEXT NOT NULL,
                        idempotency_key TEXT NOT NULL,
                        priority TEXT NOT NULL,
                        model TEXT NOT NULL,
                        endpoint TEXT NOT NULL,
                        estimated_input_tokens INTEGER NOT NULL DEFAULT 0,
                        max_output_tokens INTEGER NOT NULL DEFAULT 1024,
                        required_context_tokens INTEGER NOT NULL DEFAULT 1,
                        state TEXT NOT NULL,
                        payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                        orchestration_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                        artifacts_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                        labels_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL,
                        attempt_count INTEGER NOT NULL DEFAULT 0,
                        started_at TIMESTAMPTZ,
                        finished_at TIMESTAMPTZ,
                        next_attempt_at TIMESTAMPTZ,
                        result_json JSONB,
                        error_json JSONB,
                        UNIQUE (tenant, idempotency_key)
                    )
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE llmo_tasks
                    ADD COLUMN IF NOT EXISTS attempt_count INTEGER NOT NULL DEFAULT 0
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE llmo_tasks
                    ADD COLUMN IF NOT EXISTS next_attempt_at TIMESTAMPTZ
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS llmo_tasks_claim_idx
                    ON llmo_tasks (state, model, created_at, task_id)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS llmo_tasks_tenant_state_idx
                    ON llmo_tasks (tenant, state, created_at, task_id)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS llmo_tasks_runnable_claim_idx
                    ON llmo_tasks (state, model, next_attempt_at, created_at, task_id)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS llmo_tasks_fairness_idx
                    ON llmo_tasks (
                        tenant,
                        project,
                        service,
                        task,
                        priority,
                        model,
                        updated_at
                    )
                    WHERE attempt_count > 0
                    """
                )
            conn.commit()

    def insert_task(self, cur: Any, task: QueueTask) -> StoredTask | None:
        now = now_iso()
        task_id = f"task_{uuid4().hex}"
        cur.execute(
            f"""
            INSERT INTO llmo_tasks (
                task_id,
                tenant,
                project,
                service,
                task,
                job_id,
                idempotency_key,
                priority,
                model,
                endpoint,
                estimated_input_tokens,
                max_output_tokens,
                required_context_tokens,
                state,
                payload_json,
                orchestration_json,
                artifacts_json,
                labels_json,
                created_at,
                updated_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, 'queued', %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb,
                %s, %s
            )
            ON CONFLICT (tenant, idempotency_key) DO NOTHING
            {TASK_RETURNING_SQL}
            """,
            (
                task_id,
                task.tenant,
                task.project,
                task.service,
                task.task,
                task.job_id,
                task.idempotency_key,
                task.priority,
                task.model,
                task.endpoint,
                task.estimated_input_tokens,
                task.max_output_tokens,
                task.required_context_tokens,
                json.dumps(task.payload),
                json.dumps(task.orchestration),
                json.dumps(task.artifacts),
                json.dumps(task.labels),
                now,
                now,
            ),
        )
        row = cur.fetchone()
        return stored_task_from_row(row) if row is not None else None

    def connection(self) -> Any:
        if self.connect_factory is not None:
            return self.connect_factory()

        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError(
                "Postgres task store requires psycopg. Install psycopg[binary]."
            ) from exc
        return psycopg.connect(self.dsn, row_factory=dict_row)


def stored_task_from_row(row: Any) -> StoredTask:
    record = dict(row)
    return StoredTask.from_record(
        {
            "task_id": record.get("task_id"),
            "tenant": record.get("tenant"),
            "project": record.get("project"),
            "service": record.get("service"),
            "task": record.get("task"),
            "job_id": record.get("job_id"),
            "idempotency_key": record.get("idempotency_key"),
            "priority": record.get("priority"),
            "model": record.get("model"),
            "endpoint": record.get("endpoint"),
            "estimated_input_tokens": record.get("estimated_input_tokens", 0),
            "max_output_tokens": record.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS),
            "required_context_tokens": record.get("required_context_tokens", 1),
            "state": record.get("state") or "queued",
            "payload": json_value(record.get("payload_json"), {}),
            "orchestration": json_value(record.get("orchestration_json"), {}),
            "artifacts": json_value(record.get("artifacts_json"), {}),
            "labels": json_value(record.get("labels_json"), {}),
            "created_at": iso_value(record.get("created_at")),
            "updated_at": iso_value(record.get("updated_at")),
            "attempt_count": record.get("attempt_count", 0),
            "started_at": iso_value(record.get("started_at")),
            "finished_at": iso_value(record.get("finished_at")),
            "next_attempt_at": iso_value(record.get("next_attempt_at")),
            "result": json_value(record.get("result_json"), None),
            "error": json_value(record.get("error_json"), None),
        }
    )


def json_value(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


def iso_value(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
