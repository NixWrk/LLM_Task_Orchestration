import pytest

from queue_proxy.task_queue import (
    InMemoryTaskStore,
    JsonFileTaskStore,
    TaskProtocolError,
    context_bucket,
    context_plans_for_tasks,
    parse_task_queue_payload,
    queue_lengths_for_tasks,
)


def test_parse_task_queue_payload_requires_strict_v1_identity() -> None:
    with pytest.raises(TaskProtocolError, match="orchestration.tenant"):
        parse_task_queue_payload(
            {
                "model": "zotero-html-translate",
                "orchestration": {
                    "schema_version": "llmo.task.v1",
                    "project": "zotero",
                    "service": "zotero-html-translate-worker",
                    "task": "html_translate",
                    "priority": "batch",
                },
                "tasks": [
                    {
                        "job_id": "zotero:item:ABCD1234:source-html:ru",
                        "idempotency_key": "zotero:item:ABCD1234:source-html:ru:v1",
                    }
                ],
            }
        )


def test_task_store_scopes_idempotency_by_tenant() -> None:
    tasks = parse_task_queue_payload(
        {
            "model": "zotero-html-translate",
            "orchestration": {
                "schema_version": "llmo.task.v1",
                "tenant": "elvis",
                "project": "zotero",
                "service": "zotero-html-translate-worker",
                "task": "html_translate",
                "priority": "batch",
            },
            "tasks": [
                {
                    "job_id": "zotero:item:ABCD1234:source-html:ru",
                    "idempotency_key": "zotero:item:ABCD1234:source-html:ru:v1",
                }
            ],
        }
    )
    store = InMemoryTaskStore()

    accepted, reused = store.submit_many(tasks)
    duplicate_accepted, duplicate_reused = store.submit_many(tasks)

    assert len(accepted) == 1
    assert reused == []
    assert duplicate_accepted == []
    assert duplicate_reused == accepted
    assert store.queue_lengths_by_model() == {"zotero-html-translate": 1}


def test_parse_task_queue_payload_renders_employer_payload_template() -> None:
    tasks = parse_task_queue_payload(
        {
            "model": "zotero-html-translate",
            "endpoint": "/v1/chat/completions",
            "payload_template": {
                "model": "{{model}}",
                "messages": [
                    {"role": "system", "content": "{{system_prompt}}"},
                    {
                        "role": "user",
                        "content": "Translate {{text}} from {{artifacts.input_ref}}.",
                    },
                ],
                "max_tokens": "{{max_tokens}}",
            },
            "template_vars": {
                "system_prompt": "Translate scientific HTML to Russian.",
                "max_tokens": 999,
            },
            "orchestration": {
                "schema_version": "llmo.task.v1",
                "tenant": "elvis",
                "project": "zotero",
                "service": "zotero-html-translate-worker",
                "task": "html_translate",
                "priority": "batch",
            },
            "tasks": [
                {
                    "job_id": "zotero:item:TEMPLATE:source-html:ru",
                    "idempotency_key": "zotero:item:TEMPLATE:source-html:ru:v1",
                    "artifacts": {"input_ref": "file:///tmp/source.html"},
                    "template_vars": {
                        "text": "<p>Hello.</p>",
                        "max_tokens": 123,
                    },
                }
            ],
        }
    )

    task = tasks[0]

    assert task.payload["model"] == "zotero-html-translate"
    assert task.payload["messages"][0]["content"] == "Translate scientific HTML to Russian."
    assert task.payload["messages"][1]["content"] == (
        "Translate <p>Hello.</p> from file:///tmp/source.html."
    )
    assert task.payload["max_tokens"] == 123
    assert task.max_output_tokens == 123
    assert task.estimated_input_tokens > 0


def test_parse_task_queue_payload_rejects_unknown_template_variable() -> None:
    with pytest.raises(TaskProtocolError, match="unknown template variable"):
        parse_task_queue_payload(
            {
                "model": "zotero-html-translate",
                "payload_template": {
                    "model": "{{model}}",
                    "messages": [{"role": "user", "content": "{{missing_text}}"}],
                },
                "orchestration": {
                    "schema_version": "llmo.task.v1",
                    "tenant": "elvis",
                    "project": "zotero",
                    "service": "zotero-html-translate-worker",
                    "task": "html_translate",
                    "priority": "batch",
                },
                "tasks": [
                    {
                        "job_id": "zotero:item:TEMPLATE:source-html:ru",
                        "idempotency_key": "zotero:item:TEMPLATE:source-html:ru:v1",
                    }
                ],
            }
        )


def test_task_store_builds_context_plan_for_model_queue() -> None:
    tasks = parse_task_queue_payload(
        {
            "model": "zotero-html-translate",
            "orchestration": {
                "schema_version": "llmo.task.v1",
                "tenant": "elvis",
                "project": "zotero",
                "service": "zotero-html-translate-worker",
                "task": "html_translate",
                "priority": "batch",
                "max_parallel": 4,
                "lms_context_length": 32768,
            },
            "tasks": [
                {
                    "job_id": "zotero:item:ABCD1234:source-html:ru",
                    "idempotency_key": "zotero:item:ABCD1234:source-html:ru:v1",
                    "tokens": {
                        "estimated_input_tokens": 5200,
                        "max_output_tokens": 1200,
                    },
                },
                {
                    "job_id": "zotero:item:EFGH5678:source-html:ru",
                    "idempotency_key": "zotero:item:EFGH5678:source-html:ru:v1",
                    "tokens": {
                        "estimated_input_tokens": 9200,
                        "max_output_tokens": 1800,
                    },
                },
            ],
        }
    )
    store = InMemoryTaskStore()

    store.submit_many(tasks)
    plan = store.context_plans_by_model()["zotero-html-translate"]

    assert plan["queued_tasks"] == 2
    assert plan["max_required_context_tokens"] == 11000
    assert plan["recommended_lms_context_length"] == 16384
    assert plan["recommended_lms_parallel"] == 2
    assert plan["total_slot_context_tokens"] == 32768
    assert plan["context_cap_tokens"] == 32768
    assert plan["oversized_tasks"] == []


def test_queue_explain_helpers_use_only_queued_tasks() -> None:
    tasks = parse_task_queue_payload(
        {
            "model": "zotero-html-translate",
            "orchestration": {
                "schema_version": "llmo.task.v1",
                "tenant": "elvis",
                "project": "zotero",
                "service": "zotero-html-translate-worker",
                "task": "html_translate",
                "priority": "batch",
                "max_parallel": 4,
                "lms_context_length": 32768,
            },
            "tasks": [
                {
                    "job_id": "zotero:item:QUEUED:source-html:ru",
                    "idempotency_key": "zotero:item:QUEUED:source-html:ru:v1",
                    "tokens": {
                        "estimated_input_tokens": 5200,
                        "max_output_tokens": 1200,
                    },
                    "payload": {
                        "model": "zotero-html-translate",
                        "messages": [{"role": "user", "content": "translate"}],
                    },
                },
                {
                    "job_id": "zotero:item:DONE:source-html:ru",
                    "idempotency_key": "zotero:item:DONE:source-html:ru:v1",
                    "tokens": {
                        "estimated_input_tokens": 9200,
                        "max_output_tokens": 1800,
                    },
                    "payload": {
                        "model": "zotero-html-translate",
                        "messages": [{"role": "user", "content": "translate"}],
                    },
                },
            ],
        }
    )
    store = InMemoryTaskStore()
    accepted, _reused = store.submit_many(tasks)
    claimed = store.claim_next()
    store.record_result(claimed.task_id, {"body": {"ok": True}})

    remaining = store.list_tasks("elvis", limit=10)

    assert queue_lengths_for_tasks(remaining) == {"zotero-html-translate": 1}
    plan = context_plans_for_tasks(remaining)["zotero-html-translate"]
    assert plan["queued_tasks"] == 1
    assert plan["task_contexts"][0]["job_id"] == accepted[1].job_id


def test_json_file_task_store_persists_tasks_and_idempotency(tmp_path) -> None:
    store_path = tmp_path / "tasks.json"
    tasks = parse_task_queue_payload(
        {
            "model": "zotero-html-translate",
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
                },
                {
                    "job_id": "zotero:item:EFGH5678:source-html:ru",
                    "idempotency_key": "zotero:item:EFGH5678:source-html:ru:v1",
                    "tokens": {
                        "estimated_input_tokens": 9200,
                        "max_output_tokens": 1800,
                    },
                },
            ],
        }
    )
    first_store = JsonFileTaskStore(store_path)

    accepted, reused = first_store.submit_many(tasks)
    restarted_store = JsonFileTaskStore(store_path)
    duplicate_accepted, duplicate_reused = restarted_store.submit_many(tasks)

    assert len(accepted) == 2
    assert reused == []
    assert duplicate_accepted == []
    assert [task.task_id for task in duplicate_reused] == [
        task.task_id for task in accepted
    ]
    assert restarted_store.queue_lengths_by_model() == {"zotero-html-translate": 2}
    assert (
        restarted_store.context_plans_by_model()["zotero-html-translate"][
            "recommended_lms_parallel"
        ]
        == 2
    )


def test_task_store_status_claim_and_result_flow() -> None:
    tasks = parse_task_queue_payload(
        {
            "model": "zotero-html-translate",
            "orchestration": {
                "schema_version": "llmo.task.v1",
                "tenant": "elvis",
                "project": "zotero",
                "service": "zotero-html-translate-worker",
                "task": "html_translate",
                "priority": "batch",
            },
            "tasks": [
                {
                    "job_id": "zotero:item:ABCD1234:source-html:ru",
                    "idempotency_key": "zotero:item:ABCD1234:source-html:ru:v1",
                    "payload": {
                        "model": "zotero-html-translate",
                        "messages": [{"role": "user", "content": "translate"}],
                    },
                }
            ],
        }
    )
    store = InMemoryTaskStore()

    accepted, _reused = store.submit_many(tasks)
    task = accepted[0]
    claimed = store.claim_next()
    completed = store.record_result(task.task_id, {"body": {"ok": True}})

    assert claimed == task
    assert store.get_task("other", task.task_id) is None
    assert store.get_task("elvis", task.task_id) == task
    assert store.list_tasks("elvis", state="succeeded") == [task]
    assert completed.state == "succeeded"
    assert completed.attempt_count == 1
    assert completed.result == {"body": {"ok": True}}
    assert completed.finished_at is not None
    assert store.task_counts_by_state() == {
        (
            "elvis",
            "zotero",
            "zotero-html-translate-worker",
            "html_translate",
            "zotero-html-translate",
            "succeeded",
        ): 1
    }


def test_task_store_retry_waits_until_next_attempt_is_due() -> None:
    tasks = parse_task_queue_payload(
        {
            "model": "zotero-html-translate",
            "orchestration": {
                "schema_version": "llmo.task.v1",
                "tenant": "elvis",
                "project": "zotero",
                "service": "zotero-html-translate-worker",
                "task": "html_translate",
                "priority": "batch",
            },
            "tasks": [
                {
                    "job_id": "zotero:item:ABCD1234:source-html:ru",
                    "idempotency_key": "zotero:item:ABCD1234:source-html:ru:v1",
                    "payload": {
                        "model": "zotero-html-translate",
                        "messages": [{"role": "user", "content": "translate"}],
                    },
                }
            ],
        }
    )
    store = InMemoryTaskStore()
    accepted, _reused = store.submit_many(tasks)

    claimed = store.claim_next()
    retry = store.record_retry(
        accepted[0].task_id,
        {"type": "upstream_request_failed", "retryable": True},
        "2999-01-01T00:00:00+00:00",
    )

    assert claimed == accepted[0]
    assert retry.state == "queued"
    assert retry.attempt_count == 1
    assert retry.next_attempt_at == "2999-01-01T00:00:00+00:00"
    assert store.claim_next() is None


def test_task_store_cancel_is_tenant_scoped() -> None:
    tasks = parse_task_queue_payload(
        {
            "model": "zotero-html-translate",
            "orchestration": {
                "schema_version": "llmo.task.v1",
                "tenant": "elvis",
                "project": "zotero",
                "service": "zotero-html-translate-worker",
                "task": "html_translate",
                "priority": "batch",
            },
            "tasks": [
                {
                    "job_id": "zotero:item:ABCD1234:source-html:ru",
                    "idempotency_key": "zotero:item:ABCD1234:source-html:ru:v1",
                }
            ],
        }
    )
    store = InMemoryTaskStore()
    accepted, _reused = store.submit_many(tasks)

    assert store.cancel_task("other", accepted[0].task_id) is None
    cancelled = store.cancel_task("elvis", accepted[0].task_id)

    assert cancelled is not None
    assert cancelled.state == "cancelled"
    assert store.queue_lengths_by_model() == {}


def test_task_store_claim_treats_current_priorities_as_equal() -> None:
    batch = parse_task_queue_payload(
        {
            "model": "zotero-html-translate",
            "orchestration": {
                "schema_version": "llmo.task.v1",
                "tenant": "elvis",
                "project": "zotero",
                "service": "zotero-html-translate-worker",
                "task": "html_translate",
                "priority": "batch",
            },
            "tasks": [
                {
                    "job_id": "zotero:item:BATCH:source-html:ru",
                    "idempotency_key": "zotero:item:BATCH:source-html:ru:v1",
                }
            ],
        }
    )
    interactive = parse_task_queue_payload(
        {
            "model": "zotero-html-translate",
            "orchestration": {
                "schema_version": "llmo.task.v1",
                "tenant": "elvis",
                "project": "zotero",
                "service": "zotero-html-translate-worker",
                "task": "html_translate",
                "priority": "interactive",
            },
            "tasks": [
                {
                    "job_id": "zotero:item:INTERACTIVE:source-html:ru",
                    "idempotency_key": "zotero:item:INTERACTIVE:source-html:ru:v1",
                }
            ],
        }
    )
    store = InMemoryTaskStore()

    accepted_batch, _reused = store.submit_many(batch)
    store.submit_many(interactive)

    assert store.claim_next() == accepted_batch[0]


def test_task_store_claims_fairly_between_employers() -> None:
    tenant_a = parse_task_queue_payload(
        {
            "model": "zotero-html-translate",
            "orchestration": {
                "schema_version": "llmo.task.v1",
                "tenant": "tenant-a",
                "project": "zotero",
                "service": "zotero-html-translate-worker",
                "task": "html_translate",
                "priority": "batch",
            },
            "tasks": [
                {
                    "job_id": "tenant-a:first",
                    "idempotency_key": "tenant-a:first:v1",
                },
                {
                    "job_id": "tenant-a:second",
                    "idempotency_key": "tenant-a:second:v1",
                },
            ],
        }
    )
    tenant_b = parse_task_queue_payload(
        {
            "model": "zotero-html-translate",
            "orchestration": {
                "schema_version": "llmo.task.v1",
                "tenant": "tenant-b",
                "project": "zotero",
                "service": "zotero-html-translate-worker",
                "task": "html_translate",
                "priority": "batch",
            },
            "tasks": [
                {
                    "job_id": "tenant-b:first",
                    "idempotency_key": "tenant-b:first:v1",
                }
            ],
        }
    )
    store = InMemoryTaskStore()
    accepted_a, _reused = store.submit_many(tenant_a)
    accepted_b, _reused = store.submit_many(tenant_b)

    first = store.claim_next()
    store.record_result(first.task_id, {"body": {"ok": True}})
    second = store.claim_next()

    assert first == accepted_a[0]
    assert second == accepted_b[0]


def test_context_bucket_returns_next_bucket() -> None:
    assert context_bucket(4096) == 4096
    assert context_bucket(4097) == 8192
