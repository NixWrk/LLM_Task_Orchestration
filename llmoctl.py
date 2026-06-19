from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

from orchestrator_client import (
    DEFAULT_LIFECYCLE_URL,
    DEFAULT_QUEUE_PROXY_URL,
    OrchestratorClient,
    join_url,
)


class CliError(Exception):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}: {body}", file=sys.stderr)
        return 1
    except URLError as exc:
        print(f"Request failed: {exc.reason}", file=sys.stderr)
        return 1
    except CliError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if result is not None:
        print_json(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llmoctl")
    parser.add_argument(
        "--queue-url",
        default=os.environ.get("LLMO_QUEUE_URL", DEFAULT_QUEUE_PROXY_URL),
    )
    parser.add_argument(
        "--lifecycle-url",
        default=os.environ.get("LLMO_LIFECYCLE_URL", DEFAULT_LIFECYCLE_URL),
    )
    parser.add_argument("--api-key", default=os.environ.get("LLMO_API_KEY"))
    subparsers = parser.add_subparsers(required=True)

    models = subparsers.add_parser("models", help="List configured and dynamic models")
    models.set_defaults(func=cmd_models)

    registry = subparsers.add_parser("registry", help="List backend registry instances")
    registry.set_defaults(func=cmd_registry)

    cleanup = subparsers.add_parser("cleanup", help="Drain/stop idle backend registry instances")
    cleanup.set_defaults(func=cmd_cleanup)

    metrics = subparsers.add_parser("metrics", help="Print lifecycle Prometheus metrics")
    metrics.set_defaults(func=cmd_metrics)

    explain_plan = subparsers.add_parser(
        "explain-plan",
        help="Explain lifecycle placement for a plan file or current tenant queue",
    )
    explain_plan.add_argument("--file", help="JSON file with queue_lengths/context_plans.")
    add_tenant_arg(explain_plan)
    explain_plan.add_argument("--model")
    explain_plan.add_argument("--limit", type=int, default=500)
    explain_plan.set_defaults(func=cmd_explain_plan)

    tasks = subparsers.add_parser("tasks", help="List durable tasks for a tenant")
    add_task_query_args(tasks)
    tasks.set_defaults(func=cmd_tasks)

    task = subparsers.add_parser("task", help="Get one durable task by id")
    task.add_argument("task_id")
    add_tenant_arg(task)
    task.set_defaults(func=cmd_task)

    cancel_task = subparsers.add_parser("cancel-task", help="Cancel a durable task")
    cancel_task.add_argument("task_id")
    add_tenant_arg(cancel_task)
    cancel_task.set_defaults(func=cmd_cancel_task)

    allocate = subparsers.add_parser("allocate", help="Allocate a backend for a model")
    allocate.add_argument("model")
    add_orchestration_args(allocate)
    allocate.set_defaults(func=cmd_allocate)

    chat = subparsers.add_parser("chat", help="Send a chat completion through queue proxy")
    chat.add_argument("model")
    chat.add_argument("prompt")
    chat.add_argument("--max-tokens", type=int, default=64)
    chat.add_argument("--stream", action="store_true")
    add_orchestration_args(chat)
    chat.set_defaults(func=cmd_chat)

    embeddings = subparsers.add_parser("embeddings", help="Request embeddings through queue proxy")
    embeddings.add_argument("model")
    embeddings.add_argument("text")
    add_orchestration_args(embeddings)
    embeddings.set_defaults(func=cmd_embeddings)

    return parser


def add_orchestration_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gpu", default="auto")
    parser.add_argument("--estimated-vram-gb", type=float)
    parser.add_argument("--safety-margin-gb", type=float)
    parser.add_argument("--max-parallel", type=int)
    parser.add_argument("--max-queued-requests", type=int)
    parser.add_argument("--idle-ttl-seconds", type=int)
    parser.add_argument("--load-strategy", choices=["none", "cli", "cli-if-available"])
    parser.add_argument("--lms-gpu")
    parser.add_argument("--lms-context-length", type=int)
    parser.add_argument("--lms-ttl-seconds", type=int)
    parser.add_argument("--no-warmup", action="store_true")


def cmd_models(args: argparse.Namespace) -> Any:
    return client_from_args(args).models()


def cmd_registry(args: argparse.Namespace) -> Any:
    return client_from_args(args).registry()


def cmd_cleanup(args: argparse.Namespace) -> Any:
    return client_from_args(args).cleanup()


def cmd_metrics(args: argparse.Namespace) -> str:
    return client_from_args(args).metrics()


def cmd_explain_plan(args: argparse.Namespace) -> Any:
    client = client_from_args(args)
    if args.file:
        payload = read_json_file(args.file)
        if not isinstance(payload, dict):
            raise CliError("explain-plan --file must contain a JSON object.")
        return client.explain_plan(
            queue_lengths=payload.get("queue_lengths") or {},
            context_plans=payload.get("context_plans"),
        )
    return client.explain_tasks(
        tenant=require_tenant(args),
        model=args.model,
        limit=args.limit,
    )


def cmd_tasks(args: argparse.Namespace) -> Any:
    return client_from_args(args).list_tasks(
        tenant=require_tenant(args),
        state=args.state,
        model=args.model,
        limit=args.limit,
    )


def cmd_task(args: argparse.Namespace) -> Any:
    return client_from_args(args).get_task(args.task_id, tenant=require_tenant(args))


def cmd_cancel_task(args: argparse.Namespace) -> Any:
    return client_from_args(args).cancel_task(args.task_id, tenant=require_tenant(args))


def cmd_allocate(args: argparse.Namespace) -> Any:
    return client_from_args(args).allocate(args.model, orchestration_from_args(args))


def cmd_chat(args: argparse.Namespace) -> Any:
    return client_from_args(args).chat(
        args.model,
        args.prompt,
        max_tokens=args.max_tokens,
        stream=args.stream,
        orchestration=orchestration_from_args(args),
    )


def cmd_embeddings(args: argparse.Namespace) -> Any:
    return client_from_args(args).embeddings(
        args.model,
        args.text,
        orchestration_from_args(args),
    )


def client_from_args(args: argparse.Namespace) -> OrchestratorClient:
    return OrchestratorClient(
        queue_url=args.queue_url,
        lifecycle_url=args.lifecycle_url,
        api_key=args.api_key,
    )


def orchestration_from_args(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for attr, key in (
        ("gpu", "gpu"),
        ("estimated_vram_gb", "estimated_vram_gb"),
        ("safety_margin_gb", "safety_margin_gb"),
        ("max_parallel", "max_parallel"),
        ("max_queued_requests", "max_queued_requests"),
        ("idle_ttl_seconds", "idle_ttl_seconds"),
        ("load_strategy", "load_strategy"),
        ("lms_gpu", "lms_gpu"),
        ("lms_context_length", "lms_context_length"),
        ("lms_ttl_seconds", "lms_ttl_seconds"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            payload[key] = value
    if getattr(args, "no_warmup", False):
        payload["warmup_enabled"] = False
    return payload


def add_tenant_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tenant",
        default=(
            os.environ.get("LLMO_TENANT")
            or os.environ.get("LLM_ORCHESTRATOR_TENANT")
            or ""
        ),
        help="Tenant namespace for durable task status APIs.",
    )


def add_task_query_args(parser: argparse.ArgumentParser) -> None:
    add_tenant_arg(parser)
    parser.add_argument("--state")
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int, default=100)


def require_tenant(args: argparse.Namespace) -> str:
    tenant = str(getattr(args, "tenant", "") or "").strip()
    if not tenant:
        raise CliError("durable task commands require --tenant or LLMO_TENANT.")
    return tenant


def read_json_file(path: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise CliError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CliError(f"invalid JSON in {path}: {exc.msg}") from exc


def print_json(value: Any) -> None:
    if isinstance(value, str):
        print_text(value.rstrip())
        return
    print_text(json.dumps(value, indent=2, ensure_ascii=False))


def print_text(value: str) -> None:
    try:
        print(value)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(value.encode("utf-8", errors="replace") + b"\n")


if __name__ == "__main__":
    raise SystemExit(main())
