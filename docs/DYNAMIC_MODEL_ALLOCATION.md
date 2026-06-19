# Dynamic Model Allocation

Applications do not need to know which GPU or backend process is ready. They send an OpenAI-compatible request to queue proxy and name the model they want.

For the canonical cross-project envelope and field semantics, see
[Unified Task Protocol](UNIFIED_TASK_PROTOCOL.md). This document focuses on the
dynamic allocation path that currently implements part of that protocol.

## Chat Request

```http
POST http://localhost:4100/v1/chat/completions
Content-Type: application/json
Authorization: Bearer <QUEUE_PROXY_API_KEY if configured>
```

```json
{
  "model": "qwen3_5_9b_q6k",
  "messages": [
    {"role": "user", "content": "Return exactly: ok"}
  ],
  "max_tokens": 64,
  "orchestration": {
    "schema_version": "llmo.task.v1",
    "tenant": "elvis",
    "project": "example",
    "service": "example-worker",
    "task": "chat_completion",
    "job_id": "example:chat:001",
    "idempotency_key": "example:chat:001:v1",
    "priority": "foreground",
    "gpu": "auto",
    "max_parallel": 1,
    "max_queued_requests": 8,
    "queue_timeout_seconds": 30,
    "safety_margin_gb": 1,
    "idle_ttl_seconds": 900,
    "tokens": {
      "max_output_tokens": 512,
      "max_total_tokens": 8192
    }
  }
}
```

`orchestration` is consumed by queue proxy and lifecycle. It is removed before forwarding to LM Studio, vLLM, or another OpenAI-compatible backend.

`estimated_vram_gb` is optional for LM Studio dynamic models. When `dynamic_models.auto_vram_from_lms: true`, lifecycle reads `lms ls --json` metadata and estimates the VRAM reservation from `sizeBytes` plus context overhead. Send `orchestration.estimated_vram_gb` only when a caller has a better task-specific reservation hint.

After starting the stack, run the repeatable smoke test:

```powershell
docker compose up -d --build queue-proxy
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\smoke_dynamic_allocation.ps1 `
  -Model mistralai/ministral-3-3b
```

The request flow is:

1. Queue proxy resolves an explicit model policy, or builds a dynamic policy from `defaults`.
2. Queue proxy applies request-level limits from `orchestration`, capped by the configured policy.
3. Queue proxy asks lifecycle for a ready backend when registry routing is enabled.
4. Lifecycle checks existing ready instances.
5. If none exists, lifecycle creates an allocation from `dynamic_models`.
6. For `runtime: lmstudio`, lifecycle verifies `/v1/models`, warms the requested model, and marks the backend `ready`.
7. Queue proxy leases the backend, forwards the request, streams the response, and releases the lease.

## Allocation API

Queue proxy calls this endpoint automatically, but other internal controllers can call it directly:

```http
POST http://localhost:4300/allocations
Content-Type: application/json
```

```json
{
  "model": "qwen3_5_9b_q6k",
  "orchestration": {
    "schema_version": "llmo.task.v1",
    "tenant": "elvis",
    "project": "example",
    "service": "example-controller",
    "task": "model_warmup",
    "job_id": "warmup:qwen3_5_9b_q6k",
    "priority": "batch",
    "runtime": "lmstudio",
    "base_url": "http://host.docker.internal:1234/v1",
    "gpu": "gpu0",
    "lms_gpu": "max",
    "lms_context_length": 8192,
    "max_parallel": 1,
    "safety_margin_gb": 1,
    "idle_ttl_seconds": 900
  }
}
```

Success response:

```json
{
  "model": "qwen3_5_9b_q6k",
  "created": true,
  "decision": {
    "model": "qwen3_5_9b_q6k",
    "action": "start",
    "gpu_id": "gpu0",
    "reason": "vram_available",
    "required_vram_mb": 9216,
    "available_vram_mb": 49000
  },
  "instance": {
    "model": "qwen3_5_9b_q6k",
    "backend_model": "qwen3_5_9b_q6k",
    "runtime": "lmstudio",
    "base_url": "http://host.docker.internal:1234/v1",
    "state": "ready"
  }
}
```

If no GPU has enough free VRAM, lifecycle returns `409` with the placement decision and no instance.

## Required Settings

Dynamic allocation is enabled by:

```yaml
dynamic_models:
  enabled: true
  source: lmstudio
  auto_vram_from_lms: true
  lms_binary: lms
  registry_cleanup_ttl_seconds: 3600
  allowed_model_patterns:
    - "*"
  denied_model_patterns: []
  lifecycle:
    runtime: lmstudio
    base_url: http://host.docker.internal:1234/v1
    load_strategy: cli-if-available
    estimated_vram_gb: 8
    safety_margin_gb: 1
    min_replicas: 0
    max_replicas: 1
    idle_ttl_seconds: 900
    preferred_gpus:
      - auto
```

Inspect the active catalog:

```powershell
Invoke-RestMethod http://localhost:4300/catalog/models
```

`allowed_model_patterns` and `denied_model_patterns` use exact names or shell-style wildcards. For example:

```yaml
dynamic_models:
  enabled: true
  allowed_model_patterns:
    - qwen*
    - google/gemma-4-e2b
  denied_model_patterns:
    - "*embedding*"
```

Lifecycle only allocates a dynamic model when it is allowed by policy. With `load_strategy: none`, the model must already be visible through the LM Studio `/v1/models` catalog. With `load_strategy: cli` or `cli-if-available`, lifecycle starts from LM Studio CLI metadata and lets `lms load` make the model visible before healthcheck/warmup.

With `load_strategy: cli` or `cli-if-available`, lifecycle tries to run:

```powershell
lms load <model-key> --identifier <model-key> --yes
```

and later unloads models it actually loaded with:

```powershell
lms unload <model-key>
```

`cli-if-available` is the default for dynamic LM Studio profiles. It lets a Docker lifecycle container continue when Windows host `lms.exe` is not available inside the container, while local host runs can use real `lms load/unload`. If `lms load` reports that the identifier already exists, lifecycle treats that as a pre-existing LM Studio load and will not unload it during idle cleanup.

During reconcile, lifecycle also reads live `lms ps --json` state when the CLI is
available. Matching loads it did not create are recorded as `external` registry
records and counted as reserved GPU capacity. They are not selected as ready
backends and are not unloaded by cleanup/reload policy.

Reload decisions compare the queue context plan with the live LM Studio shape.
A load is reloaded immediately when its current context cannot fit a queued
task. Bucket-only increases are skipped when the current context is sufficient,
and non-critical shape improvements respect the model profile's
`reload_min_dwell_seconds` before reload.

`registry_cleanup_ttl_seconds` removes old `stopped` or `failed` LM Studio allocation records from the registry after the TTL. Idle ready instances are stopped first by `idle_ttl_seconds`; cleanup then purges stale records.

Queue proxy should route through lifecycle:

```text
ENABLE_BACKEND_REGISTRY_ROUTING=true
```

Use strict mode when every request must have a lifecycle-ready backend:

```text
REQUIRE_BACKEND_REGISTRY_BACKEND=true
```

With strict mode disabled, queue proxy may fall back to LiteLLM if allocation fails.

## CLI

Install the project in the local virtualenv:

```powershell
.\.venv\Scripts\pip install -e ".[dev]"
```

Then use `llmoctl` for common operations:

```powershell
llmoctl models
llmoctl registry
llmoctl allocate qwen/qwen3.5-9b --gpu auto --lms-gpu max --lms-context-length 8192
llmoctl chat qwen/qwen3.5-9b "Return exactly: ok" --max-tokens 8
llmoctl embeddings text-embedding-bge-m3 "hello"
llmoctl tasks --tenant elvis --state queued
llmoctl task task_123 --tenant elvis
llmoctl cancel-task task_123 --tenant elvis
llmoctl explain-plan --tenant elvis
llmoctl explain-plan --file .\plan.json
llmoctl cleanup
llmoctl metrics
```

Useful environment defaults:

```text
LLMO_QUEUE_URL=http://localhost:4100
LLMO_LIFECYCLE_URL=http://localhost:4300
LLMO_API_KEY=<queue-proxy-api-key>
LLMO_TENANT=elvis
```

`llmoctl explain-plan --tenant <tenant>` reads the tenant's queued durable tasks
from queue proxy and asks lifecycle why the current queue is ready, waiting for
GPU capacity, starting a backend, blocked by oversized tasks, or headed for a
reload. `--file` can be used with a JSON object containing `queue_lengths` and
optional `context_plans` to explain an arbitrary lifecycle plan.

The same APIs are available directly:

```http
GET  http://localhost:4100/tasks/explain?tenant=elvis
POST http://localhost:4300/explain-plan
```

## Cleanup And Metrics

Manual cleanup:

```http
POST http://localhost:4300/cleanup
Content-Type: application/json
```

```json
{
  "queue_lengths": {
    "qwen/qwen3.5-9b": 0
  }
}
```

Prometheus metrics are exposed by both queue proxy and lifecycle:

- Queue proxy `:4100/metrics`: `llm_queue_length`, `llm_active_requests`, request/error/latency/token-budget counters.
- Lifecycle `:4300/metrics`: `llm_backend_instances`, `llm_backend_active_requests`, `llm_backend_reserved_vram_mb`, `llm_allocations_total`, `llm_gpu_memory_total_mb`, `llm_gpu_memory_used_mb`, `llm_gpu_memory_free_mb`, `llm_gpu_inventory_up`.
