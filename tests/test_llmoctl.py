import llmoctl


def test_orchestration_from_args_maps_cli_values() -> None:
    parser = llmoctl.build_parser()
    args = parser.parse_args(
        [
            "allocate",
            "qwen/qwen3.5-9b",
            "--gpu",
            "gpu1",
            "--estimated-vram-gb",
            "9",
            "--max-parallel",
            "2",
            "--lms-gpu",
            "max",
            "--lms-context-length",
            "8192",
            "--no-warmup",
        ]
    )

    payload = llmoctl.orchestration_from_args(args)

    assert payload["gpu"] == "gpu1"
    assert payload["estimated_vram_gb"] == 9
    assert payload["max_parallel"] == 2
    assert payload["lms_gpu"] == "max"
    assert payload["lms_context_length"] == 8192
    assert payload["warmup_enabled"] is False


def test_join_url_normalizes_slashes() -> None:
    assert llmoctl.join_url("http://localhost:4100/", "/v1/models") == (
        "http://localhost:4100/v1/models"
    )


def test_build_parser_reads_url_defaults_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("LLMO_QUEUE_URL", "http://queue")
    monkeypatch.setenv("LLMO_LIFECYCLE_URL", "http://lifecycle")

    args = llmoctl.build_parser().parse_args(["registry"])

    assert args.queue_url == "http://queue"
    assert args.lifecycle_url == "http://lifecycle"


def test_streaming_chat_uses_text_request(monkeypatch) -> None:
    calls = []

    class FakeClient:
        def chat(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "data: ok"

    monkeypatch.setattr(llmoctl, "client_from_args", lambda _args: FakeClient())
    args = llmoctl.build_parser().parse_args(
        [
            "--queue-url",
            "http://queue",
            "--api-key",
            "sk-test",
            "chat",
            "qwen",
            "hello",
            "--stream",
        ]
    )

    result = llmoctl.cmd_chat(args)

    assert result == "data: ok"
    assert calls[0][0] == ("qwen", "hello")
    assert calls[0][1]["stream"] is True


def test_task_commands_call_tenant_scoped_client_methods(monkeypatch) -> None:
    calls = []

    class FakeClient:
        def list_tasks(self, **kwargs):
            calls.append(("list_tasks", kwargs))
            return {"tasks": []}

        def get_task(self, task_id, **kwargs):
            calls.append(("get_task", task_id, kwargs))
            return {"task_id": task_id}

        def cancel_task(self, task_id, **kwargs):
            calls.append(("cancel_task", task_id, kwargs))
            return {"task_id": task_id, "state": "cancelled"}

    monkeypatch.setattr(llmoctl, "client_from_args", lambda _args: FakeClient())

    tasks_args = llmoctl.build_parser().parse_args(
        [
            "tasks",
            "--tenant",
            "elvis",
            "--state",
            "queued",
            "--model",
            "local-main",
            "--limit",
            "25",
        ]
    )
    task_args = llmoctl.build_parser().parse_args(["task", "task_123", "--tenant", "elvis"])
    cancel_args = llmoctl.build_parser().parse_args(
        ["cancel-task", "task_123", "--tenant", "elvis"]
    )

    assert llmoctl.cmd_tasks(tasks_args) == {"tasks": []}
    assert llmoctl.cmd_task(task_args) == {"task_id": "task_123"}
    assert llmoctl.cmd_cancel_task(cancel_args)["state"] == "cancelled"
    assert calls[0] == (
        "list_tasks",
        {
            "tenant": "elvis",
            "state": "queued",
            "model": "local-main",
            "limit": 25,
        },
    )
    assert calls[1] == ("get_task", "task_123", {"tenant": "elvis"})
    assert calls[2] == ("cancel_task", "task_123", {"tenant": "elvis"})


def test_task_commands_require_tenant(monkeypatch) -> None:
    monkeypatch.delenv("LLMO_TENANT", raising=False)
    monkeypatch.delenv("LLM_ORCHESTRATOR_TENANT", raising=False)
    args = llmoctl.build_parser().parse_args(["tasks"])

    try:
        llmoctl.cmd_tasks(args)
    except llmoctl.CliError as exc:
        assert "require --tenant" in str(exc)
    else:
        raise AssertionError("cmd_tasks should require tenant")
