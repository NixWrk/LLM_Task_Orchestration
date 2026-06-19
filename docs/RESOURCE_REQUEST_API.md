# Resource Request API

This document describes the lower-level lifecycle and capacity APIs. Client
projects should treat [Unified Task Protocol](UNIFIED_TASK_PROTOCOL.md) as the
canonical contract for submitting LLM work to the orchestrator.

The short version:

- normal inference requests go to `queue-proxy` on `/v1/...`;
- capacity and startup requests go to the lifecycle control plane;
- clients should request intent and constraints, not manually reserve GPU memory unless they intentionally override the scheduler hint;
- the orchestrator decides placement, replica count, backend URL, and routing.
- dynamic model requests must pass the lifecycle allow/deny policy.

## Current Implemented Flow

Today the implemented API is split across two services.

For the concrete request format now supported by queue proxy and lifecycle `POST /allocations`, see [Dynamic Model Allocation](DYNAMIC_MODEL_ALLOCATION.md).

For the canonical cross-project request envelope, concurrency semantics, future
persistent task queue, and OCR GPU coordination rules, see
[Unified Task Protocol](UNIFIED_TASK_PROTOCOL.md).

### 1. Ask Lifecycle To Plan Or Reconcile

Lifecycle can plan or create dry-run backend instances based on queue pressure:

```http
POST http://localhost:4300/plan
Content-Type: application/json
```

```json
{
  "queue_lengths": {
    "local-main": 1
  }
}
```

To create planned backend registry entries:

```http
POST http://localhost:4300/reconcile
Content-Type: application/json
```

```json
{
  "queue_lengths": {
    "local-main": 1
  }
}
```

Current response shape:

```json
{
  "dry_run": true,
  "gpu_count": 2,
  "models": [
    {
      "model": "local-main",
      "ready_replicas": 0,
      "active_replicas": 0,
      "desired_replicas": 1,
      "decisions": [
        {
          "model": "local-main",
          "action": "start",
          "gpu_id": "gpu0",
          "reason": "vram_available",
          "required_vram_mb": 9216,
          "available_vram_mb": 22528
        }
      ]
    }
  ],
  "created_instances": []
}
```

### 2. Send Inference To Queue Proxy

Clients use the queue proxy as an OpenAI-compatible endpoint:

```http
POST http://localhost:4100/v1/chat/completions
Authorization: Bearer <key>
Content-Type: application/json
```

```json
{
  "model": "local-main",
  "messages": [
    {
      "role": "user",
      "content": "Return exactly: ok"
    }
  ],
  "temperature": 0,
  "max_tokens": 8
}
```

The queue proxy enforces:

- per-model concurrency;
- per-model queue size;
- queue timeout;
- input/output/total token budgets;
- optional routing through lifecycle backend registry.

## Capacity Allocation API

Lifecycle now exposes `POST /allocations` for dynamic LM Studio/OpenAI-compatible model allocation. The broader contract below is the long-form shape for service/task-aware reservations; unsupported fields should currently be treated as future metadata unless they also appear in [Dynamic Model Allocation](DYNAMIC_MODEL_ALLOCATION.md).

### Request Capacity

```http
POST http://localhost:4300/allocations
Content-Type: application/json
Authorization: Bearer <admin-or-service-key>
```

```json
{
  "task_id": "ocr-batch-2026-05-29-001",
  "service": "zotero-worker",
  "model": "qwen-14b",
  "priority": "batch",
  "mode": "shared",
  "resources": {
    "min_replicas": 1,
    "max_replicas": 2,
    "gpu_count": 1,
    "preferred_gpus": ["gpu0", "gpu1"],
    "estimated_vram_gb": 16,
    "safety_margin_gb": 2,
    "max_active_requests": 2,
    "max_queued_requests": 64,
    "max_input_tokens": 8192,
    "max_output_tokens": 2048,
    "max_total_tokens": 10240,
    "idle_ttl_seconds": 900,
    "ttl_seconds": 7200
  },
  "runtime": {
    "type": "vllm",
    "artifact": "D:/models/qwen-14b",
    "image": "vllm/vllm-openai:latest",
    "volumes": [
      {
        "host_path": "D:/models/qwen-14b",
        "container_path": "/models/qwen-14b",
        "mode": "ro"
      }
    ],
    "environment": {
      "HF_HOME": "/root/.cache/huggingface"
    },
    "extra_args": ["--max-model-len", "8192"]
  },
  "warmup": {
    "enabled": true,
    "prompt": "Return exactly: ok",
    "max_tokens": 8
  }
}
```

Important semantics:

- `mode: "shared"` means reuse compatible ready backends when possible.
- `mode: "dedicated"` means create capacity reserved for this task or service.
- `preferred_gpus` is a constraint, not a command. The scheduler may reject if VRAM is insufficient.
- `estimated_vram_gb` is a reservation hint. Actual runtime memory should be measured and fed back later.
- `ttl_seconds` bounds how long the allocation may live.
- `idle_ttl_seconds` controls automatic scale-down after no active requests.

### Capacity Response

```json
{
  "allocation_id": "alloc_01jz_resource_001",
  "task_id": "ocr-batch-2026-05-29-001",
  "service": "zotero-worker",
  "state": "starting",
  "model": "qwen-14b",
  "client_endpoint": "http://localhost:4100/v1",
  "client_model": "qwen-14b",
  "expires_at": "2026-05-29T14:30:00Z",
  "limits": {
    "max_active_requests": 2,
    "max_queued_requests": 64,
    "max_input_tokens": 8192,
    "max_output_tokens": 2048,
    "max_total_tokens": 10240
  },
  "backends": [
    {
      "instance_id": "qwen-14b-gpu0-a1b2c3d4",
      "state": "starting",
      "gpu_ids": ["gpu0"],
      "reserved_vram_mb": 18432,
      "base_url": null
    }
  ]
}
```

The client should poll allocation status until it becomes `ready`.

### Allocation Status

```http
GET http://localhost:4300/allocations/alloc_01jz_resource_001
```

```json
{
  "allocation_id": "alloc_01jz_resource_001",
  "state": "ready",
  "client_endpoint": "http://localhost:4100/v1",
  "client_model": "qwen-14b",
  "backends": [
    {
      "instance_id": "qwen-14b-gpu0-a1b2c3d4",
      "state": "ready",
      "gpu_ids": ["gpu0"],
      "active_requests": 0,
      "base_url": "http://host.docker.internal:8100/v1"
    }
  ]
}
```

Expected states:

```text
requested -> planning -> starting -> warming -> ready -> draining -> released
                                      |
                                    failed
```

### Release Capacity

```http
DELETE http://localhost:4300/allocations/alloc_01jz_resource_001
```

Expected behavior:

1. Mark allocation as `draining`.
2. Stop routing new requests to dedicated backends.
3. Wait for `active_requests == 0`.
4. Stop runtime containers/processes.
5. Free reserved VRAM.
6. Mark allocation as `released`.

## Inference After Allocation

Once allocation state is `ready`, the client sends normal OpenAI-compatible requests:

```http
POST http://localhost:4100/v1/chat/completions
Authorization: Bearer <service-key>
Content-Type: application/json
X-Service-ID: zotero-worker
X-Task-ID: ocr-batch-2026-05-29-001
X-Priority: batch
```

```json
{
  "model": "qwen-14b",
  "messages": [
    {
      "role": "user",
      "content": "Extract structured metadata from this OCR text..."
    }
  ],
  "max_tokens": 1024
}
```

Future queue proxy behavior should use these headers for:

- per-service quotas;
- priority queues;
- audit logs;
- allocation-aware routing.

## Error Responses

Use stable JSON errors:

```json
{
  "error": {
    "type": "insufficient_gpu_vram",
    "message": "No allowed GPU has enough free VRAM for qwen-14b.",
    "details": {
      "required_vram_mb": 18432,
      "best_available_vram_mb": 12288
    }
  }
}
```

Recommended error types:

- `insufficient_gpu_vram`
- `model_policy_not_found`
- `runtime_adapter_unavailable`
- `runtime_start_failed`
- `warmup_failed`
- `allocation_expired`
- `queue_full`
- `queue_timeout`
- `token_budget_exceeded`
- `no_ready_backend`

## CLI Recommendation

The repository now includes a thin operational CLI:

```text
llmoctl
```

Implemented commands:

```powershell
llmoctl models
llmoctl registry
llmoctl allocate qwen/qwen3.5-9b --gpu auto --lms-gpu max --lms-context-length 8192
llmoctl chat qwen/qwen3.5-9b "Return exactly: ok" --max-tokens 8
llmoctl chat qwen/qwen3.5-9b "stream" --stream
llmoctl embeddings text-embedding-bge-m3 "hello"
llmoctl tasks --tenant elvis --state queued
llmoctl task task_123 --tenant elvis
llmoctl cancel-task task_123 --tenant elvis
llmoctl explain-plan --tenant elvis
llmoctl explain-plan --file .\plan.json
llmoctl cleanup
llmoctl metrics
```

The CLI should be a thin wrapper over the HTTP API. It should not contain scheduler logic.

## MCP Recommendation

MCP is also useful, but it should come after the HTTP API and CLI are stable.

Good MCP tools:

- `list_gpu_inventory`
- `list_backend_instances`
- `request_llm_capacity`
- `get_allocation_status`
- `release_llm_capacity`
- `run_llm_smoke_test`
- `explain_scheduler_decision`

MCP is best for agents, IDE automation, and workflow systems that need to ask for capacity without shell scripting. It should call the same lifecycle/queue proxy API as the CLI.

Recommended order:

1. Done: stabilize the HTTP allocation and lifecycle APIs.
2. Done: add the Python client package.
3. Done: add `llmoctl` CLI.
4. Next: add an MCP server as an adapter over the same client package if agent
   workflows need it.

## Implemented Low-Level API

The full task-aware allocation object in this document remains the long-form
resource contract. The repository currently implements these production
building blocks:

- Lifecycle `POST /plan`
- Lifecycle `POST /explain-plan`
- Lifecycle `POST /reconcile`
- Lifecycle `POST /cleanup`
- Lifecycle `POST /allocations`
- Lifecycle `GET /registry`
- Queue proxy `POST /tasks/queue`
- Queue proxy `GET /tasks`
- Queue proxy `GET /tasks/{task_id}`
- Queue proxy `DELETE /tasks/{task_id}`
- Queue proxy `GET /tasks/explain`
- Queue proxy `/v1/...` OpenAI-compatible inference routes
- Lifecycle `POST /registry/{instance_id}/lease`
- Lifecycle `DELETE /registry/{instance_id}/lease`

The next implementation step is to add persistent allocation IDs and service/task ownership on top of the implemented dynamic model allocation path.

## Zotero HTML Translation Contract

`zotero-html-translate-worker` is an ordinary OpenAI-compatible client. It owns
HTML segmentation, prompts, quality checks, and result files, but it does not
start, load, unload, or discover LM Studio models. It sends translation calls to
the queue proxy:

```http
POST http://localhost:4100/v1/chat/completions
```

The request names the desired model/profile and passes task hints in
`orchestration`:

```json
{
  "model": "p6_google_gemma-4-26b-a4b@q6_k",
  "messages": [{"role": "user", "content": "..."}],
  "temperature": 0.0,
  "max_tokens": 1024,
  "stream": false,
  "orchestration": {
    "schema_version": "llmo.task.v1",
    "tenant": "elvis",
    "project": "zotero",
    "service": "zotero-html-translate-worker",
    "task": "html_translate",
    "job_id": "zotero:item:ABCD1234:source-html:ru",
    "idempotency_key": "zotero:item:ABCD1234:source-html:ru:v1",
    "priority": "batch",
    "lms_context_length": 32768,
    "estimated_vram_gb": 20,
    "idle_ttl_seconds": 120,
    "lms_ttl_seconds": 120
  }
}
```

Queue proxy strips `orchestration` before forwarding to the backend. Lifecycle
uses it only as bounded allocation metadata, so other projects can use the same
contract with their own `project`, `task`, model, and resource hints.
