import pytest

from queue_proxy.task_queue import (
    InMemoryTaskStore,
    TaskProtocolError,
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
