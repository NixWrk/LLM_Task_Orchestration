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
