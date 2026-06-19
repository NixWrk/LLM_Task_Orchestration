# Unified Task Contract Plan

Last updated: 2026-06-19

## Purpose

This orchestrator is the single entry point for local LLM work from all Elvis
projects. Client projects should not start LM Studio, choose GPUs, unload
models, or implement their own model queue. They submit a task request with the
model/profile they need, task metadata, and resource hints; the orchestrator owns
queueing, concurrency, backend lifecycle, GPU placement, and observability.

## Canonical Request Shape

All OpenAI-compatible inference calls should keep the ordinary OpenAI payload
and add a single `orchestration` object. The object is consumed by queue proxy
and lifecycle and must be stripped before forwarding to LM Studio, vLLM, SGLang,
or any other OpenAI-compatible backend.

```http
POST http://localhost:4100/v1/chat/completions
Authorization: Bearer <service-key>
Content-Type: application/json
X-Project-ID: zotero
X-Service-ID: zotero-html-translate-worker
X-Task-ID: zotero:item:ABCD1234:translate:ru
X-Request-ID: <uuid-or-deterministic-id>
X-Priority: batch
```

```json
{
  "model": "p6_google_gemma-4-26b-a4b@q6_k",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "temperature": 0.0,
  "top_p": 1.0,
  "max_tokens": 8192,
  "stream": false,
  "orchestration": {
    "schema_version": "2026-06-19",
    "project": "zotero",
    "service": "zotero-html-translate-worker",
    "task": "html_translate",
    "job_id": "zotero:item:ABCD1234:source-html:ru",
    "idempotency_key": "zotero:item:ABCD1234:source-html:ru:v1",
    "priority": "batch",
    "runtime": "lmstudio",
    "gpu": "auto",
    "model_profile": "scientific-translation-ru",
    "lms_context_length": 32768,
    "estimated_vram_gb": 20,
    "max_parallel": 1,
    "max_queued_requests": 64,
    "queue_timeout_seconds": 7200,
    "idle_ttl_seconds": 900,
    "ttl_seconds": 7200,
    "tokens": {
      "max_input_tokens": 32768,
      "max_output_tokens": 8192,
      "max_total_tokens": 40960
    },
    "artifacts": {
      "input_ref": "file:///data/html/source/ABCD1234.html",
      "output_ref": "file:///data/html/ru/ABCD1234.html"
    }
  }
}
```

### Required Fields

1. `model`: public model/profile name requested by the caller.
2. `messages` or other OpenAI-compatible input for the endpoint.
3. `orchestration.schema_version`.
4. `orchestration.project`.
5. `orchestration.service`.
6. `orchestration.task`.
7. `orchestration.job_id`.
8. `orchestration.priority`.

### Optional Resource Hints

1. `runtime`: `lmstudio`, `vllm`, `sglang`, or `openai-compatible`.
2. `gpu`: `auto` or a scheduler-visible GPU id.
3. `estimated_vram_gb`.
4. `max_parallel`.
5. `max_queued_requests`.
6. `queue_timeout_seconds`.
7. `idle_ttl_seconds`.
8. `ttl_seconds`.
9. `tokens`.
10. `artifacts`.

Hints are bounded by server-side policy. A client can ask for more parallelism
or a longer queue timeout, but the orchestrator decides the admitted values.

## Concurrency Rule

`max_parallel` is a requested active-request upper bound, not a proof that the
backend can generate that many responses at once.

Example: if four translation jobs are submitted and the queue proxy shows
`active_requests=1` and `queued_requests=3`, the system has four admitted client
jobs but only one active model generation. This is expected when the selected
backend, model profile, or GPU policy allows only one active request.

The orchestrator must expose both values separately:

1. submitted/admitted jobs;
2. active model generations;
3. queued jobs waiting for the model;
4. backend replica count;
5. rejected jobs and rejection reason.

A project must not infer real generation parallelism from the number of worker
containers or client processes it started.

## Persistent Queue Target

The current queue proxy is sufficient for admission control, but the target
state is a durable queue owned by the orchestrator:

1. `POST /v1/...` continues to support synchronous OpenAI-compatible calls.
2. A future `POST /tasks` accepts the same canonical envelope for long jobs.
3. `GET /tasks/{task_id}` returns durable status, queue position, active backend,
   timing, token usage, and final result metadata.
4. `DELETE /tasks/{task_id}` cancels queued work or marks running work as
   draining/cancel-requested when supported by the backend.
5. Queue state survives orchestrator restarts.
6. Idempotency keys prevent duplicate work when callers retry after timeouts.

This keeps client projects simple: they submit durable work and poll status
instead of implementing their own LLM retry queue.

## GPU Coordination With OCR

OCR is also a GPU consumer. The LLM orchestrator remains the owner of LLM model
lifecycle, but it should expose a general GPU capacity view so OCR and LLM work
do not blindly compete for VRAM.

Target behavior:

1. OCR service advertises queued/running jobs and coarse VRAM needs.
2. LLM lifecycle reads GPU inventory and external reservations before starting
   or scaling model backends.
3. OCR can request a GPU slot or check whether a GPU-heavy LLM backend should be
   drained before a large OCR job.
4. Both services publish metrics with `project`, `service`, `task`, and `gpu`.
5. Server-side policies decide whether batch translation or OCR has priority.

This does not mean OCR should send OCR work to the LLM queue. OCR keeps its own
job API and queue, while both services coordinate through shared GPU inventory,
reservations, and documented priority rules.

## Elvis Projects Layout

The working directory should group repositories by role:

```text
D:/Elvis_projects/
  Zotero_Automation/
    Zotero_automatization/
    zotero-ingest-worker/
    zotero-html-translate-worker/
    zotero-file-relay/
  Surya_Chandra_PDF_OCR/
  LLM_Orchestrator/
```

`Surya_Chandra_PDF_OCR` and `LLM_Orchestrator` are universal infrastructure
projects. Large product projects, such as Zotero automation, live under their
own project folder and call those infrastructure services through their public
APIs.

## Implementation Tasks

1. Add schema validation for the `orchestration` object.
