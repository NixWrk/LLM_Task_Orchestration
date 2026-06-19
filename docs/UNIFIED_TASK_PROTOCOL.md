# Unified Task Protocol

Protocol version: `llmo.task.v1`

Last updated: 2026-06-19

Implementation roadmap: [Task Context Orchestration Implementation Plan](TASK_CONTEXT_ORCHESTRATION_PLAN.md).

## Purpose

This protocol is the contract between client projects and the local LLM
orchestrator.

Client containers, workers, scripts, and agents describe the LLM work they need:
tenant, model, task identity, priority, token limits, runtime hints, and
GPU/resource hints. The orchestrator owns queue admission, model policy, backend
selection, GPU placement, LM Studio/vLLM/SGLang startup, backend leases, cleanup,
and metrics.

Clients must not:

1. Start or stop LM Studio, vLLM, or SGLang directly.
2. Pick a backend URL from the registry.
3. Lease or release backend instances directly.
4. Infer available generation parallelism from their own worker count.
5. Treat GPU hints as hard reservations unless the orchestrator confirms them.

Clients, also called employers in the scheduling model, own the task prompt and
payload. If a task needs a prompt, the employer must submit it as an
OpenAI-compatible `payload` or an explicit task template. The orchestrator
executes accepted work and manages resources; it must not invent prompts for
employer-specific tasks.

## Entry Points

### Synchronous Inference

Current stable path:

```http
POST http://localhost:4100/v1/chat/completions
POST http://localhost:4100/v1/responses
POST http://localhost:4100/v1/completions
POST http://localhost:4100/v1/embeddings
```

The request body is an ordinary OpenAI-compatible payload plus one
`orchestration` object. Queue proxy consumes `orchestration` and strips it before
forwarding the request to LM Studio, vLLM, SGLang, LiteLLM, or another
OpenAI-compatible backend.

Most clients should start here. Queue proxy can allocate a dynamic backend
through lifecycle when registry routing is enabled.

### Capacity Warmup

Optional current path:

```http
POST http://localhost:4300/allocations
```

Use this only when a controller wants to warm a model before the first
inference request. Ordinary clients should let queue proxy allocate on demand.

The request shape is:

```json
{
  "model": "zotero-html-translate",
  "orchestration": {
    "schema_version": "llmo.task.v1",
    "tenant": "elvis",
    "project": "zotero",
    "service": "zotero-html-translate-worker",
    "task": "html_translate",
    "job_id": "warmup:zotero-html-translate",
    "priority": "batch",
    "gpu": "auto",
    "lms_gpu": "max",
    "lms_context_length": 32768,
    "max_parallel": 1,
    "idle_ttl_seconds": 900
  }
}
```

### Durable Tasks

Current queue-submission path:

```http
POST http://localhost:4100/tasks/queue
```

This accepts a batch of tenant-scoped tasks, records them in the task store, and
asks lifecycle to reconcile capacity from the resulting per-model queue lengths.
It is the first implemented step toward durable task execution.

Target path, not implemented yet:

```http
POST http://localhost:4100/tasks
GET http://localhost:4100/tasks/{task_id}
DELETE http://localhost:4100/tasks/{task_id}
```

Durable tasks will accept the same envelope, persist queue state, support
idempotency across restarts, expose queue position, and let clients poll instead
of holding a long HTTP request open. The synchronous `/v1/...` contract remains
valid after durable tasks are added.

Durable tasks are tenant-scoped. Tasks from different tenants must not share task
ids, idempotency records, status listings, quotas, or priority accounting.

### Queue Submission

Example HTML translation queue:

```json
{
  "model": "zotero-html-translate",
  "endpoint": "/v1/chat/completions",
  "payload_template": {
    "model": "{{model}}",
    "messages": [
      {"role": "system", "content": "{{system_prompt}}"},
      {"role": "user", "content": "Translate to Russian:\n\n{{html}}"}
    ],
    "temperature": 0,
    "max_tokens": "{{max_tokens}}"
  },
  "template_vars": {
    "system_prompt": "Translate scientific HTML to Russian."
  },
  "orchestration": {
    "schema_version": "llmo.task.v1",
    "tenant": "elvis",
    "project": "zotero",
    "service": "zotero-html-translate-worker",
    "task": "html_translate",
    "priority": "batch",
    "gpu": "auto",
    "lms_gpu": "max",
    "lms_context_length": 32768,
    "max_parallel": 1,
    "idle_ttl_seconds": 900
  },
  "tasks": [
    {
      "job_id": "zotero:item:ABCD1234:source-html:ru",
      "idempotency_key": "zotero:item:ABCD1234:source-html:ru:v1",
      "tokens": {
        "estimated_input_tokens": 5200,
        "max_output_tokens": 1200
      },
      "artifacts": {
        "input_ref": "file:///data/zotero/ABCD1234/02.en.polish.html",
        "output_ref": "file:///data/zotero/ABCD1234/03.ru.translate.html"
      },
      "template_vars": {
        "html": "<p>...</p>",
        "max_tokens": 1200
      }
    },
    {
      "job_id": "zotero:item:EFGH5678:source-html:ru",
      "idempotency_key": "zotero:item:EFGH5678:source-html:ru:v1",
      "tokens": {
        "estimated_input_tokens": 9200,
        "max_output_tokens": 1800
      },
      "artifacts": {
        "input_ref": "file:///data/zotero/EFGH5678/02.en.polish.html",
        "output_ref": "file:///data/zotero/EFGH5678/03.ru.translate.html"
      },
      "template_vars": {
        "html": "<p>...</p>",
        "max_tokens": 1800
      }
    }
  ]
}
```

Successful response:

```json
{
  "accepted_tasks": 2,
  "reused_tasks": 0,
  "queue_lengths": {
    "zotero-html-translate": 2
  },
  "context_plans": {
    "zotero-html-translate": {
      "queued_tasks": 2,
      "max_required_context_tokens": 11000,
      "recommended_lms_context_length": 16384,
      "requested_parallel": 4,
      "recommended_lms_parallel": 2,
      "total_slot_context_tokens": 32768,
      "context_cap_tokens": 32768,
      "oversized_tasks": []
    }
  },
  "tasks": [
    {
      "task_id": "task_...",
      "tenant": "elvis",
      "project": "zotero",
      "service": "zotero-html-translate-worker",
      "task": "html_translate",
      "job_id": "zotero:item:ABCD1234:source-html:ru",
      "idempotency_key": "zotero:item:ABCD1234:source-html:ru:v1",
      "priority": "batch",
      "model": "zotero-html-translate",
      "endpoint": "/v1/chat/completions",
      "estimated_input_tokens": 5200,
      "max_output_tokens": 1200,
      "required_context_tokens": 6400,
      "state": "queued",
      "attempt_count": 0,
      "next_attempt_at": null,
      "reused": false
    }
  ],
  "capacity": {
    "state": "reconciled",
    "result": {
      "models": [
        {
          "model": "zotero-html-translate",
          "desired_replicas": 1,
          "decisions": [
            {
              "action": "start",
              "gpu_id": "gpu0",
              "reason": "vram_available"
            }
          ]
        }
      ]
    }
  }
}
```

The queue submission endpoint does not make client containers choose GPUs. It
only reports the queue. Lifecycle calculates desired replicas and GPU placement
from the queue length, model policy, registry reservations, and GPU inventory.

`context_plans` is the orchestrator's first-pass packing plan. A client may send
per-task token estimates, but it does not decide slot count or context length.
The orchestrator rounds each model's maximum required context into a context
bucket, chooses the requested number of active slots from queue length and policy
hints, and reports whether any task is too large for the declared context cap.

`payload_template` is an employer-owned OpenAI payload template. The
orchestrator only renders `{{variable}}` placeholders from common
`template_vars`, each task's `template_vars`, plus standard task metadata such
as `model`, `job_id`, `idempotency_key`, `artifacts`, and `labels`; it does not
invent prompts. Per-task variables override common variables. If a template
references an unknown variable, queue admission fails with
`invalid_task_protocol`.

## Canonical Request

Example chat request:

```http
POST http://localhost:4100/v1/chat/completions
Authorization: Bearer <queue-proxy-service-key>
Content-Type: application/json
X-Tenant-ID: elvis
X-Project-ID: zotero
X-Service-ID: zotero-html-translate-worker
X-Task-ID: zotero:item:ABCD1234:source-html:ru
X-Request-ID: 018fe4d5-8f2c-7c4e-8b6f-c4ad0c2e6e39
```

```json
{
  "model": "zotero-html-translate",
  "messages": [
    {"role": "system", "content": "Translate scientific HTML to Russian."},
    {"role": "user", "content": "<html>...</html>"}
  ],
  "temperature": 0.0,
  "top_p": 1.0,
  "max_tokens": 8192,
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
    "runtime": "lmstudio",
    "gpu": "auto",
    "model_profile": "scientific-translation-ru",
    "max_parallel": 1,
    "max_queued_requests": 64,
    "queue_timeout_seconds": 900,
    "startup_timeout_seconds": 1800,
    "idle_ttl_seconds": 900,
    "ttl_seconds": 7200,
    "estimated_vram_gb": 20,
    "safety_margin_gb": 1,
    "load_strategy": "cli-if-available",
    "lms_gpu": "max",
    "lms_context_length": 32768,
    "lms_parallel": 1,
    "lms_ttl_seconds": 3600,
    "tokens": {
      "max_input_tokens": 32768,
      "max_output_tokens": 8192,
      "max_total_tokens": 40960
    },
    "artifacts": {
      "input_ref": "file:///data/zotero/html/source/ABCD1234.html",
      "output_ref": "file:///data/zotero/html/ru/ABCD1234.html"
    },
    "labels": {
      "language": "ru",
      "domain": "scientific_html"
    }
  }
}
```

`model` is the public orchestrator model/profile name. It is not necessarily the
backend model id. The orchestrator may rewrite it to `backend_model` before
forwarding to the backend.

## Required Fields

For new clients, these fields are required:

| Field | Meaning |
| --- | --- |
| `model` | Public model or profile requested by the client. |
| OpenAI input | Endpoint-specific input such as `messages`, `input`, or `prompt`. |
| `orchestration.schema_version` | Must be `llmo.task.v1`. |
| `orchestration.tenant` | Required tenant/employer namespace. |
| `orchestration.project` | Product or project namespace, for example `zotero`. |
| `orchestration.service` | Calling service or worker name. |
| `orchestration.task` | Stable task type, for example `html_translate`. |
| `orchestration.job_id` | Caller-visible job id for logs and status. |
| `orchestration.priority` | Scheduling class. |

The protocol is strict: new clients must send every required field. A request
that claims `schema_version: "llmo.task.v1"` but omits required metadata should
be rejected with a stable validation error. Temporary compatibility for legacy
requests may exist in code, but it is not part of this protocol.

## Headers

Headers duplicate the most important body metadata for HTTP logs, reverse
proxies, and non-JSON tooling. The body is authoritative.

| Header | Body equivalent |
| --- | --- |
| `X-Tenant-ID` | `orchestration.tenant` |
| `X-Project-ID` | `orchestration.project` |
| `X-Service-ID` | `orchestration.service` |
| `X-Task-ID` | `orchestration.job_id` |
| `X-Request-ID` | unique HTTP attempt id |
| `X-Idempotency-Key` | `orchestration.idempotency_key` |
| `X-Priority` | `orchestration.priority` |

`X-Request-ID` changes on each HTTP attempt. `idempotency_key` remains stable
across retries for the same logical work item.

## Tenant Isolation

`tenant` is the top-level isolation boundary. It answers the question: "who owns
this work?" Internal Elvis projects can use `tenant: "elvis"`. If this
orchestrator later serves several teams, customers, or products with independent
quotas, each one gets its own tenant id.

The orchestrator must scope these records by tenant:

1. durable task ids;
2. idempotency keys;
3. task status listings;
4. queue position and priority accounting;
5. per-tenant quotas and rate limits;
6. audit logs and metrics;
7. artifact references and result metadata.

Recommended uniqueness rules:

| Value | Uniqueness scope |
| --- | --- |
| `job_id` | Unique inside `(tenant, project, service)` while active. |
| `idempotency_key` | Unique inside `tenant`. |
| durable `task_id` | Globally unique, but every lookup still checks tenant ownership. |

Queue fairness is currently calculated by
`(tenant, project, service, task, priority, model)`. That makes the employer
group explicit and prevents one tenant/service/task batch from silently
consuming every due slot before another employer group gets a turn.

## Priority Classes

Use one of:

| Priority | Intended use |
| --- | --- |
| `interactive` | Human waiting on the result. Small queues, short timeouts. |
| `foreground` | User-visible automation that can wait briefly. |
| `batch` | Background jobs such as translation or indexing. |
| `maintenance` | Cleanup, reprocessing, low-urgency quality passes. |

Priority is a scheduling hint. Server-side policy decides the actual queue
order, maximum concurrency, and whether a request is admitted.

Current policy treats all employers as equal priority. The priority field stays
in the protocol so future policy can distinguish interactive, foreground, batch,
and maintenance classes without changing the task envelope.

## Orchestration Fields

### Identity

| Field | Type | Notes |
| --- | --- | --- |
| `schema_version` | string | `llmo.task.v1`. |
| `tenant` | string | Required tenant/employer namespace. |
| `project` | string | Required namespace. |
| `service` | string | Required caller id. |
| `task` | string | Required task type. |
| `job_id` | string | Required caller job id. |
| `idempotency_key` | string | Strongly recommended. Required for durable tasks. |
| `priority` | string | Required priority class. |
| `labels` | object | Optional low-cardinality labels for audit and metrics. |

### Model And Runtime

| Field | Type | Notes |
| --- | --- | --- |
| `runtime` | string | `lmstudio`, `vllm`, `sglang`, or `openai-compatible`. |
| `model_profile` | string | Optional semantic profile label. Routing still uses top-level `model`. |
| `load_strategy` | string | `none`, `cli`, or `cli-if-available` for LM Studio. |
| `base_url` | string | Admin/controller hint only. Ordinary clients should omit it. |

### GPU And Capacity

| Field | Type | Notes |
| --- | --- | --- |
| `gpu` | string or array | `auto` or scheduler-visible ids like `gpu0`. |
| `preferred_gpus` | array | Equivalent explicit list. `gpu` is shorter for clients. |
| `estimated_vram_gb` | number | Reservation hint. Policy and runtime metadata may override it. |
| `safety_margin_gb` | number | Extra VRAM headroom. |
| `max_parallel` | integer | Requested active generation limit. Bounded by model policy. |
| `max_active_requests` | integer | Alias for `max_parallel`. |
| `max_queued_requests` | integer | Requested queue depth. Bounded by model policy. |
| `queue_timeout_seconds` | number | Requested queue wait limit. Bounded by model policy. |
| `startup_timeout_seconds` | number | Model/backend startup wait limit. |
| `idle_ttl_seconds` | integer | Stop ready idle backend after this many seconds. |
| `ttl_seconds` | integer | Upper bound for task/allocation lifetime. |

### LM Studio Hints

| Field | Type | Notes |
| --- | --- | --- |
| `lms_gpu` | string | Usually `max`; passed to `lms load` when available. |
| `lms_context_length` | integer | Context length requested for the loaded model. |
| `lms_parallel` | integer | LM Studio parallel slot count. Defaults from `max_parallel` if omitted. |
| `lms_ttl_seconds` | integer | LM Studio model TTL hint. |
| `lms_binary` | string | Admin/controller override. Ordinary clients should omit it. |

### Token Limits

| Field | Type | Notes |
| --- | --- | --- |
| `tokens.default_max_output_tokens` | integer | Default output cap when request omits one. |
| `tokens.max_input_tokens` | integer | Estimated input token limit. |
| `tokens.max_output_tokens` | integer | Output token limit. |
| `tokens.max_total_tokens` | integer | Input estimate plus output budget. |

The same token fields may also be sent at the top level of `orchestration` for
legacy compatibility. New clients should prefer `orchestration.tokens`.

### Artifacts

| Field | Type | Notes |
| --- | --- | --- |
| `artifacts.input_ref` | string | Optional URI/path to the source artifact. |
| `artifacts.output_ref` | string | Optional URI/path where caller will store output. |
| `artifacts.manifest_ref` | string | Optional URI/path to richer task metadata. |

The orchestrator does not need to read artifacts for synchronous `/v1/...`
inference. They exist for audit, durable tasks, and future retries.

## Semantics

### Strict Validation

For `llmo.task.v1`, validation is mandatory before queue admission or lifecycle
allocation. The orchestrator should reject:

1. missing required identity fields;
2. unsupported `schema_version`;
3. unknown priority class;
4. invalid numeric resource hints;
5. malformed `tokens`, `artifacts`, or `labels`;
6. a mismatch between required headers and body fields when both are present.

Body fields are authoritative. Header mismatches should be treated as client
bugs, not silently corrected.

### Context Planning

Workers describe task size; the orchestrator decides the backend shape.

For every queued task, the orchestrator should know or estimate:

1. input tokens;
2. output token budget;
3. required context tokens;
4. target model/profile;
5. priority and tenant.

For each model queue, the orchestrator calculates:

1. maximum required context among active queued tasks;
2. recommended context bucket, such as `8192`, `16384`, or `32768`;
3. recommended parallel slots from queue length, priority, and policy;
4. total slot context budget;
5. tasks that cannot fit inside the declared context cap.

For LM Studio, this maps to `lms_context_length` and `lms_parallel`. The model is
still loaded once; parallel slots let multiple requests share that load. More
slots do not mean `model_vram * slots`, but larger context and active generations
can increase KV/runtime memory and reduce tokens per second.

Lifecycle must compare the current live LM Studio state with the planned future
shape. It must distinguish lifecycle-owned loads from pre-existing external
loads. Owned loads may be drained and reloaded by policy; unowned loads are
treated as reserved external capacity unless an explicit takeover policy is
enabled.

### Graceful Reload

If the current LM Studio load does not match the queue's context plan, lifecycle
should use a graceful reload policy:

1. mark the backend `draining`;
2. stop routing new tasks to that backend;
3. wait until `active_requests == 0`;
4. unload the LM Studio identifier only if lifecycle owns that load;
5. reload with the planned `lms_context_length` and `lms_parallel`;
6. warm up and mark the backend `ready`;
7. resume routing queued tasks.

Reload should not happen for every small queue change. Use context buckets,
minimum dwell time, and hysteresis so the system does not unload/reload the same
model repeatedly while a batch is arriving.

If the live LM Studio context can satisfy the largest currently queued task,
lifecycle may keep the existing load even when the configured target bucket is
larger. Non-critical improvements, such as increasing parallel slots, should
wait for the configured minimum dwell time before reload. Hard mismatches, such
as a live context that is too small for a queued task, remain reload candidates.

VRAM planning should use both current live state and future estimates such as
`lms load --estimate-only`. The scheduler should account for model size, planned
context, planned parallel slots, active owned loads, and unowned external loads.

### Policy Wins

Client fields are intent and bounded hints. The orchestrator may clamp or reject
them based on server-side policy.

Examples:

1. A client requests `max_parallel: 4`, but model policy allows `1`; effective
   active generation limit is `1`.
2. A client requests `max_output_tokens: 8192`, but policy allows `4096`; queue
   proxy may clamp or reject depending on model policy.
3. A client requests `gpu: gpu0`, but no allowed GPU has enough VRAM; lifecycle
   returns an allocation failure or queue proxy returns `503 no_ready_backend`.

### Concurrency

`max_parallel` means requested active model generations, not submitted jobs.

If a worker starts eight HTTP requests and model policy allows one active
request, queue proxy should report one active request and seven queued requests.
That is correct. Worker count is not generation parallelism.

### Idempotency

For synchronous `/v1/...` requests, clients should retry only when the HTTP
attempt failed before a trustworthy response was received. Current code does not
persist idempotency results.

For future durable `/tasks`, `idempotency_key` will be mandatory. Repeating the
same key must return the existing task instead of creating duplicate work.

### Streaming

`stream: true` keeps OpenAI streaming semantics. Queue proxy still owns the
backend lease until the upstream stream closes or the client disconnects.

`stream: false` returns a buffered upstream response. Queue proxy still releases
the limiter slot and backend lease when the response finishes or the client
disconnects.

## Request Flow

For synchronous inference:

1. Client sends OpenAI-compatible request to queue proxy with `orchestration`.
2. Queue proxy validates JSON, resolves public model policy, and applies bounded
   request overrides.
3. Queue proxy estimates input tokens and enforces token limits.
4. Queue proxy admits the request into the per-model limiter queue.
5. Queue proxy asks lifecycle registry for a ready backend.
6. If no backend exists and dynamic allocation is enabled, lifecycle starts or
   loads one according to policy.
7. Lifecycle chooses GPU placement, records reserved VRAM, verifies/warmups the
   backend, and marks it `ready`.
8. Queue proxy leases the backend instance.
9. Queue proxy strips `orchestration`, rewrites `model` to `backend_model` when
   needed, and forwards to the backend.
10. Queue proxy releases the backend lease and limiter slot on completion,
    upstream error, or client disconnect.
11. Lifecycle cleanup stops idle dynamic backends after TTL policy allows it.

## Response Contract

Successful synchronous calls return the backend's OpenAI-compatible response.
Queue proxy may add orchestrator headers such as:

| Header | Meaning |
| --- | --- |
| `x-llm-output-tokens-capped` | Output budget was reduced to fit policy. |

Future headers should use the `x-llmo-` prefix, for example:

| Header | Meaning |
| --- | --- |
| `x-llmo-request-id` | Effective request id. |
| `x-llmo-backend-instance-id` | Backend instance used. |
| `x-llmo-queue-wait-ms` | Time spent waiting in queue. |
| `x-llmo-effective-max-parallel` | Effective active generation limit. |

## Error Contract

Errors should use this JSON shape:

```json
{
  "error": {
    "type": "queue_timeout",
    "message": "Timed out waiting for an available model request slot.",
    "details": {
      "model": "zotero-html-translate",
      "queue_timeout_seconds": 900
    }
  }
}
```

Current error responses may omit `details`. Clients must key behavior from
`error.type`, not message text.

Stable error types:

| Error type | Meaning |
| --- | --- |
| `invalid_json` | Request body is not a JSON object. |
| `model_policy_not_found` | Requested model is not configured or allowed. |
| `token_budget_exceeded` | Input/output/total token policy failed. |
| `queue_full` | Per-model queue is full. |
| `queue_timeout` | Request waited too long for an active slot. |
| `no_ready_backend` | No backend could be allocated or selected. |
| `insufficient_gpu_vram` | Scheduler could not place the model. |
| `runtime_adapter_unavailable` | Requested runtime cannot be controlled. |
| `runtime_start_failed` | Backend process/container failed to start. |
| `warmup_failed` | Backend did not pass warmup. |
| `invalid_task_payload` | Durable task payload is not executable for its endpoint. |
| `upstream_request_failed` | Backend request failed after routing. |
| `client_disconnected` | Client disconnected before completion. |
| `allocation_expired` | Durable allocation or task exceeded TTL. |

## Durable Task Target

Future `POST /tasks` should accept:

```json
{
  "endpoint": "/v1/chat/completions",
  "payload": {
    "model": "zotero-html-translate",
    "messages": [{"role": "user", "content": "..."}],
    "max_tokens": 8192,
    "stream": false,
    "orchestration": {
      "schema_version": "llmo.task.v1",
      "tenant": "elvis",
      "project": "zotero",
      "service": "zotero-html-translate-worker",
      "task": "html_translate",
      "job_id": "zotero:item:ABCD1234:source-html:ru",
      "idempotency_key": "zotero:item:ABCD1234:source-html:ru:v1",
      "priority": "batch"
    }
  }
}
```

Task states:

```text
submitted -> queued -> allocating -> starting -> warming -> running -> succeeded
                                                        |          |
                                                        |          -> failed
                                                        -> failed

submitted/queued/running -> cancelling -> cancelled
queued/running -> expired
```

`GET /tasks/{task_id}` should return:

1. task identity and state;
2. queue position and timing;
3. effective model policy and token budget;
4. backend instance id and GPU ids when assigned;
5. final OpenAI response or artifact refs;
6. stable error object if failed;
7. retry/idempotency metadata.

Durable task execution is orchestrator-owned after admission. Employers submit a
task once, receive task ids, and poll status. The orchestrator decides whether a
failure is retryable, when to retry, and when to mark the task permanently
failed.

### Task And Execution Characteristics

The employer owns the task description. These fields describe what should be
done:

1. `tenant`, `project`, `service`, and `task`;
2. `job_id` and `idempotency_key`;
3. `model` and `endpoint`;
4. OpenAI-compatible `payload`, or employer-owned `payload_template` plus
   `template_vars`, including the prompt when the task needs one;
5. `artifacts` and `labels`;
6. token estimates such as `estimated_input_tokens` and `max_output_tokens`;
7. requested hints such as `priority`, `gpu`, `max_parallel`, and
   `lms_context_length`.

The orchestrator owns the execution characteristics. These fields explain how
the accepted task is being or was executed:

1. `state`, `created_at`, `updated_at`, `started_at`, and `finished_at`;
2. `attempt_count`, `next_attempt_at`, and the final retry decision;
3. selected backend instance and GPU ids when known;
4. effective context/parallel plan for the model queue;
5. final response body, usage, or artifact references;
6. stable `error.type`, `retryable`, `attempt_count`, and `max_attempts` when
   execution fails.

Retryable executor failures return to `queued` with `next_attempt_at`.
Permanent failures, or retryable failures after the configured attempt limit,
end in `failed`.

The durable executor validates endpoint-specific OpenAI-compatible input before
routing. For example, `/v1/chat/completions` tasks must include
`payload.messages`, `/v1/responses` tasks must include `payload.input` or
`payload.messages`, `/v1/completions` tasks must include `payload.prompt`, and
`/v1/embeddings` tasks must include `payload.input`. Non-executable worker
metadata is rejected as `invalid_task_payload` with `retryable: false`.

When a queue request includes `payload_template`, admission renders it into the
stored task `payload`. A string that is exactly `{{name}}` preserves the
variable's JSON type; a placeholder inside a longer string is rendered as text.
This lets employers send numeric values such as `max_tokens` without turning
them into strings.

## Client Rules

Every client project should:

1. Send LLM inference only to queue proxy, not directly to LM Studio or vLLM.
2. Put stable task metadata in `orchestration`.
3. Set the correct `tenant`; do not reuse another tenant's namespace.
4. Use `model` as the public profile name, not a private backend URL.
5. Use `gpu: auto` unless it has a strong reason to constrain placement.
6. Treat all resource fields as requested hints, not guaranteed capacity.
7. Set realistic HTTP timeouts. Long batch work should migrate to durable tasks
   when `/tasks` is implemented.
8. Use `idempotency_key` for retryable logical work.
9. Handle `429`, `503`, and `413` as normal scheduling/policy outcomes.
10. Export its own job metrics with the same `tenant`, `project`, `service`, and
   `task` names used in `orchestration`.
11. Provide the prompt/payload or explicit prompt template needed to execute its
   task. Do not rely on the orchestrator to infer domain-specific prompts from
   artifact names.

## Server Rules

The orchestrator should:

1. Validate `orchestration` shape before using it for policy decisions.
2. Strip `orchestration` before forwarding to any backend.
3. Clamp caller hints to configured policy or reject when policy requires it.
4. Scope durable state, idempotency, task lists, quotas, and fairness by tenant.
5. Keep queue slots and backend leases balanced on success, error, timeout, and
   client disconnect.
6. Keep model startup and unload decisions inside lifecycle.
7. Publish metrics labelled by `model`, and eventually by `tenant`, `project`,
   `service`, `task`, `priority`, and `gpu`.
8. Record enough status to explain why a request was queued, rejected, placed,
   or failed.
9. Execute accepted durable tasks according to server-side retry and scheduling
   policy.
10. Use employer-provided payloads/templates for task execution instead of
   inventing prompts.

## Durable Storage

The task protocol must not expose the storage engine. Clients talk to HTTP APIs;
the orchestrator hides whether durable state lives in Postgres, SQLite, files, or
another backend.

The implementation should use a small storage interface, for example
`TaskStore`, with operations like:

1. create or return task by `(tenant, idempotency_key)`;
2. update task state and timestamps;
3. claim the next runnable task by
   `(tenant, project, service, task, priority, model)`;
4. record allocation/backend assignment;
5. store final response metadata or artifact refs;
6. list tasks only inside the caller's tenant;
7. expire or cancel tasks safely.

Recommended first production storage is Postgres because it is already present
in the compose stack and supports durable queues, indexes, transactions, and
multi-process workers. A SQLite or JSON-file implementation may be useful for
local development, but it must implement the same `TaskStore` contract so client
behavior does not change.

Current local durable mode uses `JsonFileTaskStore`, enabled by setting
`TASK_STORE_PATH` for queue proxy. When unset, queue proxy uses the in-memory
store for ephemeral development and tests.

Production compose mode can use `PostgresTaskStore` by setting
`TASK_STORE_BACKEND=postgres` and `TASK_STORE_DSN`. The store creates the
initial task table, indexes, and `llmo_schema_metadata` on startup. Startup
records `task_store_schema_version` and refuses to run when the database schema
is newer than the code. Formal migration tooling is still a next step.

Minimum durable task fields:

| Field | Notes |
| --- | --- |
| `task_id` | Globally unique orchestrator id. |
| `tenant` | Required isolation boundary. |
| `project`, `service`, `task`, `job_id` | Caller identity. |
| `idempotency_key` | Unique inside tenant. |
| `priority`, `model`, `endpoint` | Scheduling fields. |
| `payload_json` | Original OpenAI-compatible payload. |
| `state` | Durable task state. |
| `queue_position` | Derived or cached status field. |
| `backend_instance_id`, `gpu_ids` | Assignment once known. |
| `created_at`, `updated_at`, `started_at`, `finished_at`, `expires_at` | Timing. |
| `attempt_count`, `next_attempt_at` | Retry scheduling and status metadata. |
| `result_json` or `artifact_refs` | Final result metadata. |
| `error_json` | Stable error object on failure. |

## Current Implementation Mapping

Implemented now:

1. OpenAI-compatible `/v1/...` queue proxy.
2. `orchestration` stripping before upstream forwarding.
3. Bounded overrides for queue limits and token limits.
4. Strict `POST /tasks/queue` batch submission for tenant-scoped task queues.
5. `TaskStore` abstraction with in-memory and JSON-file implementations for
   queue/capacity checks.
6. Queue submission triggers lifecycle reconcile from per-model queue lengths.
7. Lifecycle `/plan` and `/reconcile` consume `context_plans` and report
   desired backend shape plus `reload` decisions when an LM Studio load is too
   small for the queue.
8. Dynamic model policy from `dynamic_models`.
9. Lifecycle `POST /allocations` for dynamic LM Studio/OpenAI-compatible
   backend allocation.
10. Backend registry lookup, lease, and release.
11. LM Studio hints such as `lms_gpu`, `lms_context_length`, `lms_parallel`, and
   `lms_ttl_seconds`.
12. Idle cleanup for dynamic LM Studio allocations.
13. LM Studio CLI inspection/estimate parsing for `lms ps --json` and
   `lms load --estimate-only`.
14. Initial graceful reload state machine: drain active loads, reload idle owned
   loads, and skip unowned/pre-existing LM Studio loads.
15. Tenant-scoped task status APIs:
   `GET /tasks`, `GET /tasks/{task_id}`, and `DELETE /tasks/{task_id}`.
16. Optional durable task executor enabled by `TASK_EXECUTOR_ENABLED`; it claims
   queued tasks with stored OpenAI-compatible payloads, routes through backend
   resolver, records results/errors, and releases backend leases.
17. Postgres task-store backend selected by `TASK_STORE_BACKEND=postgres` and
   `TASK_STORE_DSN`.
18. Lifecycle uses live `lms ps --json` shape before reload decisions and can
   use `lms load --estimate-only` for planned VRAM.
19. Durable task retry metadata and executor retry policy:
   `attempt_count`, `next_attempt_at`, retryable transient backend errors, and
   permanent failure for invalid/non-executable tasks.
20. Equal-priority task claiming across
   `(tenant, project, service, task, priority, model)` employer groups.
21. Durable task metrics on `/metrics` for lifecycle events, task errors,
    current task states, queue wait time, and execution duration.
22. Durable executor validates endpoint-specific OpenAI-compatible payloads
    before routing and marks non-executable worker metadata as
    `invalid_task_payload`.
23. Queue admission can render employer-provided `payload_template` objects with
    per-task `template_vars` into stored OpenAI-compatible payloads.
24. `llmoctl tasks`, `llmoctl task <id>`, and `llmoctl cancel-task <id>` expose
    tenant-scoped durable task inspection and cancellation from the operator CLI.
25. Lifecycle persists live LM Studio reconciliation metadata, represents
    matching unowned LM Studio loads as `external` registry records, reserves
    their estimated VRAM, and applies reload hysteresis before non-critical
    shape changes.
26. Lifecycle `POST /explain-plan`, queue proxy `GET /tasks/explain`, and
    `llmoctl explain-plan` expose machine-readable placement/reload explanations
    for current tenant queues or supplied plan JSON.
27. Postgres task store startup records and checks `task_store_schema_version`
    in `llmo_schema_metadata`.
28. Synchronous `/v1/...` request preparation validates declared
    `schema_version: llmo.task.v1` envelopes and rejects malformed identity,
    priority, token/resource hint, artifact, label, and GPU fields as
    `invalid_task_protocol`.

Needed next:

1. Expanded metrics/log labels from task metadata for lifecycle, reloads, GPU
   placement, and backend ownership.
2. Formal Postgres migration tooling and real-container integration tests.
3. Task execution logs and richer operator-facing task status history.
4. Zotero HTML queue submission with employer-provided payloads/prompts and
   artifact references.
5. Allocation ids and task ownership in lifecycle.
6. External GPU reservation API for non-LLM consumers such as OCR.
7. More Python client helpers that build the canonical envelope.
