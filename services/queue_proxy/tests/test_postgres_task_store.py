from __future__ import annotations

from typing import Any

from queue_proxy.postgres_task_store import (
    TASK_STORE_SCHEMA_KEY,
    TASK_STORE_SCHEMA_VERSION,
    PostgresTaskStore,
)
from queue_proxy.task_queue import parse_task_queue_payload


def test_postgres_task_store_submits_and_reuses_by_tenant_idempotency() -> None:
    database = FakePostgresDatabase()
    store = PostgresTaskStore(
        "postgresql://test",
        connect_factory=database.connect,
    )
    tasks = sample_tasks()

    accepted, reused = store.submit_many(tasks)
    duplicate_accepted, duplicate_reused = store.submit_many(tasks)

    assert database.schema_initialized is True
    assert database.metadata[TASK_STORE_SCHEMA_KEY] == str(TASK_STORE_SCHEMA_VERSION)
    assert len(accepted) == 2
    assert reused == []
    assert duplicate_accepted == []
    assert [task.task_id for task in duplicate_reused] == [
        task.task_id for task in accepted
    ]
    assert store.queue_lengths_by_model() == {"local-main": 2}


def test_postgres_task_store_rejects_newer_schema_version() -> None:
    database = FakePostgresDatabase()
    database.metadata[TASK_STORE_SCHEMA_KEY] = str(TASK_STORE_SCHEMA_VERSION + 1)

    try:
        PostgresTaskStore(
            "postgresql://test",
            connect_factory=database.connect,
        )
    except RuntimeError as exc:
        assert "schema is newer" in str(exc)
    else:
        raise AssertionError("PostgresTaskStore should reject a newer schema.")


def test_postgres_task_store_rejects_invalid_schema_version() -> None:
    database = FakePostgresDatabase()
    database.metadata[TASK_STORE_SCHEMA_KEY] = "future-ish"

    try:
        PostgresTaskStore(
            "postgresql://test",
            connect_factory=database.connect,
        )
    except RuntimeError as exc:
        assert "Invalid Postgres task store schema version" in str(exc)
    else:
        raise AssertionError("PostgresTaskStore should reject an invalid schema version.")


def test_postgres_task_store_claim_result_and_context_plan() -> None:
    database = FakePostgresDatabase()
    store = PostgresTaskStore(
        "postgresql://test",
        connect_factory=database.connect,
    )
    accepted, _reused = store.submit_many(sample_tasks())

    claimed = store.claim_next(model="local-main")
    completed = store.record_result(claimed.task_id, {"body": {"ok": True}})
    plan = store.context_plans_by_model()["local-main"]

    assert claimed.task_id == accepted[0].task_id
    assert claimed.attempt_count == 1
    assert completed.state == "succeeded"
    assert completed.result == {"body": {"ok": True}}
    assert completed.finished_at is not None
    assert plan["queued_tasks"] == 1
    assert plan["recommended_lms_context_length"] == 16384
    assert store.task_counts_by_state() == {
        (
            "elvis",
            "zotero",
            "zotero-html-translate-worker",
            "html_translate",
            "local-main",
            "succeeded",
        ): 1,
        (
            "elvis",
            "zotero",
            "zotero-html-translate-worker",
            "html_translate",
            "local-main",
            "queued",
        ): 1,
    }


def test_postgres_task_store_enforces_tenant_scoped_status_and_cancel() -> None:
    database = FakePostgresDatabase()
    store = PostgresTaskStore(
        "postgresql://test",
        connect_factory=database.connect,
    )
    accepted, _reused = store.submit_many(sample_tasks())
    task_id = accepted[0].task_id

    assert store.get_task("other", task_id) is None
    assert store.cancel_task("other", task_id) is None

    cancelled = store.cancel_task("elvis", task_id)

    assert cancelled is not None
    assert cancelled.state == "cancelled"
    assert store.get_task("elvis", task_id).state == "cancelled"
    assert store.list_tasks("elvis", state="queued") == [accepted[1]]


def test_postgres_task_store_records_retry_and_skips_future_attempt() -> None:
    database = FakePostgresDatabase()
    store = PostgresTaskStore(
        "postgresql://test",
        connect_factory=database.connect,
    )
    accepted, _reused = store.submit_many(sample_tasks())

    claimed = store.claim_next(model="local-main")
    retry = store.record_retry(
        claimed.task_id,
        {"type": "upstream_request_failed", "retryable": True},
        "2999-01-01T00:00:00+00:00",
    )
    next_claimed = store.claim_next(model="local-main")

    assert retry.state == "queued"
    assert retry.attempt_count == 1
    assert retry.next_attempt_at == "2999-01-01T00:00:00+00:00"
    assert next_claimed.task_id == accepted[1].task_id


def sample_tasks():
    return parse_task_queue_payload(
        {
            "model": "local-main",
            "orchestration": {
                "schema_version": "llmo.task.v1",
                "tenant": "elvis",
                "project": "zotero",
                "service": "zotero-html-translate-worker",
                "task": "html_translate",
                "priority": "batch",
                "max_parallel": 4,
            },
            "tasks": [
                {
                    "job_id": "zotero:item:ABCD1234:source-html:ru",
                    "idempotency_key": "zotero:item:ABCD1234:source-html:ru:v1",
                    "tokens": {
                        "estimated_input_tokens": 5200,
                        "max_output_tokens": 1200,
                    },
                    "payload": {
                        "model": "local-main",
                        "messages": [{"role": "user", "content": "translate one"}],
                    },
                },
                {
                    "job_id": "zotero:item:EFGH5678:source-html:ru",
                    "idempotency_key": "zotero:item:EFGH5678:source-html:ru:v1",
                    "tokens": {
                        "estimated_input_tokens": 9200,
                        "max_output_tokens": 1800,
                    },
                    "payload": {
                        "model": "local-main",
                        "messages": [{"role": "user", "content": "translate two"}],
                    },
                },
            ],
        }
    )


class FakePostgresDatabase:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.metadata: dict[str, str] = {}
        self.schema_initialized = False

    def connect(self) -> FakeConnection:
        return FakeConnection(self)


class FakeConnection:
    def __init__(self, database: FakePostgresDatabase) -> None:
        self.database = database
        self.commits = 0

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def cursor(self) -> FakeCursor:
        return FakeCursor(self.database)

    def commit(self) -> None:
        self.commits += 1


class FakeCursor:
    def __init__(self, database: FakePostgresDatabase) -> None:
        self.database = database
        self.rows: list[dict[str, Any]] = []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        normalized = " ".join(sql.lower().split())
        if (
            normalized.startswith("create table")
            or normalized.startswith("create index")
            or normalized.startswith("alter table")
        ):
            self.database.schema_initialized = True
            self.rows = []
            return
        if normalized.startswith("select value from llmo_schema_metadata"):
            value = self.database.metadata.get(params[0])
            self.rows = [] if value is None else [{"value": value}]
            return
        if normalized.startswith("insert into llmo_schema_metadata"):
            key, value, _updated_at = params
            self.database.metadata[str(key)] = str(value)
            self.rows = []
            return
        if normalized.startswith("insert into llmo_tasks"):
            self.insert_task(params)
            return
        if normalized.startswith("select model, count(*)"):
            self.count_by_model(params[0])
            return
        if normalized.startswith("select tenant, project, service, task, model, state"):
            self.count_by_task_state()
            return
        if normalized.startswith("select") and "state = any" in normalized:
            self.rows = self.active_rows(params[0])
            return
        if (
            normalized.startswith("select")
            and "where tenant = %s and idempotency_key = %s" in normalized
        ):
            tenant, idempotency_key = params
            self.rows = [
                row
                for row in self.database.rows.values()
                if row["tenant"] == tenant
                and row["idempotency_key"] == idempotency_key
            ]
            return
        if (
            normalized.startswith("select")
            and "where tenant = %s and task_id = %s" in normalized
        ):
            tenant, task_id = params
            row = self.database.rows.get(task_id)
            self.rows = [row] if row is not None and row["tenant"] == tenant else []
            return
        if normalized.startswith("select"):
            self.list_tasks(normalized, params)
            return
        if normalized.startswith("with group_scores") or normalized.startswith("with next_task"):
            self.claim_next(params)
            return
        if normalized.startswith("update llmo_tasks") and "state = 'succeeded'" in normalized:
            result_json, finished_at, updated_at, task_id = params
            row = self.database.rows.get(task_id)
            if row is None:
                self.rows = []
                return
            row.update(
                {
                    "state": "succeeded",
                    "result_json": result_json,
                    "error_json": None,
                    "next_attempt_at": None,
                    "finished_at": finished_at,
                    "updated_at": updated_at,
                }
            )
            self.rows = [row]
            return
        if normalized.startswith("update llmo_tasks") and "state = 'failed'" in normalized:
            error_json, finished_at, updated_at, task_id = params
            row = self.database.rows.get(task_id)
            if row is None:
                self.rows = []
                return
            row.update(
                {
                    "state": "failed",
                    "error_json": error_json,
                    "next_attempt_at": None,
                    "finished_at": finished_at,
                    "updated_at": updated_at,
                }
            )
            self.rows = [row]
            return
        if normalized.startswith("update llmo_tasks") and "state = 'queued'" in normalized:
            error_json, next_attempt_at, updated_at, task_id = params
            row = self.database.rows.get(task_id)
            if row is None:
                self.rows = []
                return
            row.update(
                {
                    "state": "queued",
                    "error_json": error_json,
                    "next_attempt_at": next_attempt_at,
                    "updated_at": updated_at,
                }
            )
            self.rows = [row]
            return
        if normalized.startswith("update llmo_tasks") and "state = 'cancelled'" in normalized:
            self.cancel_task(params)
            return
        raise AssertionError(f"Unhandled SQL: {sql}")

    def fetchone(self) -> dict[str, Any] | None:
        if not self.rows:
            return None
        return self.rows[0]

    def fetchall(self) -> list[dict[str, Any]]:
        return self.rows

    def insert_task(self, params: tuple[Any, ...]) -> None:
        tenant = str(params[1])
        idempotency_key = str(params[6])
        for row in self.database.rows.values():
            if row["tenant"] == tenant and row["idempotency_key"] == idempotency_key:
                self.rows = []
                return

        row = {
            "task_id": params[0],
            "tenant": params[1],
            "project": params[2],
            "service": params[3],
            "task": params[4],
            "job_id": params[5],
            "idempotency_key": params[6],
            "priority": params[7],
            "model": params[8],
            "endpoint": params[9],
            "estimated_input_tokens": params[10],
            "max_output_tokens": params[11],
            "required_context_tokens": params[12],
            "state": "queued",
            "payload_json": params[13],
            "orchestration_json": params[14],
            "artifacts_json": params[15],
            "labels_json": params[16],
            "created_at": params[17],
            "updated_at": params[18],
            "attempt_count": 0,
            "started_at": None,
            "finished_at": None,
            "next_attempt_at": None,
            "result_json": None,
            "error_json": None,
        }
        self.database.rows[str(row["task_id"])] = row
        self.rows = [row]

    def count_by_model(self, states: list[str]) -> None:
        counts: dict[str, int] = {}
        for row in self.database.rows.values():
            if row["state"] in states:
                counts[row["model"]] = counts.get(row["model"], 0) + 1
        self.rows = [
            {"model": model, "count": count}
            for model, count in sorted(counts.items())
        ]

    def count_by_task_state(self) -> None:
        counts: dict[tuple[str, str, str, str, str, str], int] = {}
        for row in self.database.rows.values():
            key = (
                row["tenant"],
                row["project"],
                row["service"],
                row["task"],
                row["model"],
                row["state"],
            )
            counts[key] = counts.get(key, 0) + 1
        self.rows = [
            {
                "tenant": key[0],
                "project": key[1],
                "service": key[2],
                "task": key[3],
                "model": key[4],
                "state": key[5],
                "count": count,
            }
            for key, count in sorted(counts.items())
        ]

    def active_rows(self, states: list[str]) -> list[dict[str, Any]]:
        return sorted(
            [
                row
                for row in self.database.rows.values()
                if row["state"] in states
            ],
            key=lambda row: (row["model"], row["created_at"], row["task_id"]),
        )

    def list_tasks(self, normalized_sql: str, params: tuple[Any, ...]) -> None:
        tenant = params[0]
        index = 1
        state = None
        model = None
        if "state = %s" in normalized_sql:
            state = params[index]
            index += 1
        if "model = %s" in normalized_sql:
            model = params[index]
            index += 1
        limit = int(params[index])
        rows = [
            row
            for row in self.database.rows.values()
            if row["tenant"] == tenant
            and (state is None or row["state"] == state)
            and (model is None or row["model"] == model)
        ]
        self.rows = sorted(rows, key=lambda row: (row["created_at"], row["task_id"]))[:limit]

    def claim_next(self, params: tuple[Any, ...]) -> None:
        (
            states,
            model,
            _model_again,
            now,
            _states_again,
            _model_third,
            _model_fourth,
            _now_again,
            started_at,
            updated_at,
        ) = params
        candidates = [
            row
            for row in self.database.rows.values()
            if row["state"] in states
            and (model is None or row["model"] == model)
            and (
                row.get("next_attempt_at") is None
                or str(row["next_attempt_at"]) <= str(now)
            )
        ]
        candidates.sort(key=self.claim_sort_key)
        if not candidates:
            self.rows = []
            return
        row = candidates[0]
        row["state"] = "running"
        row["started_at"] = row["started_at"] or started_at
        row["attempt_count"] += 1
        row["next_attempt_at"] = None
        row["updated_at"] = updated_at
        self.rows = [row]

    def cancel_task(self, params: tuple[Any, ...]) -> None:
        finished_at, updated_at, tenant, task_id = params
        row = self.database.rows.get(task_id)
        if (
            row is None
            or row["tenant"] != tenant
            or row["state"] in {"succeeded", "failed", "cancelled"}
        ):
            self.rows = []
            return
        row["state"] = "cancelled"
        row["finished_at"] = row["finished_at"] or finished_at
        row["next_attempt_at"] = None
        row["updated_at"] = updated_at
        self.rows = [row]

    def claim_sort_key(self, row: dict[str, Any]) -> tuple[bool, str, str, tuple[str, ...], str]:
        group = self.group_key(row)
        last_claimed_at = None
        for candidate in self.database.rows.values():
            if self.group_key(candidate) != group or candidate["attempt_count"] < 1:
                continue
            if last_claimed_at is None or candidate["updated_at"] > last_claimed_at:
                last_claimed_at = candidate["updated_at"]
        return (
            last_claimed_at is not None,
            last_claimed_at or "",
            row["created_at"],
            group,
            row["task_id"],
        )

    def group_key(self, row: dict[str, Any]) -> tuple[str, ...]:
        return (
            str(row["tenant"]),
            str(row["project"]),
            str(row["service"]),
            str(row["task"]),
            str(row["priority"]),
            str(row["model"]),
        )
