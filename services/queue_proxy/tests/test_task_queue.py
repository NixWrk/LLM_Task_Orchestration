import pytest

from queue_proxy.task_queue import (
    InMemoryTaskStore,
    TaskProtocolError,
    context_bucket,
    parse_task_queue_payload,
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


def test_context_bucket_returns_next_bucket() -> None:
    assert context_bucket(4096) == 4096
    assert context_bucket(4097) == 8192
