# Task Context Orchestration Implementation Plan

Last updated: 2026-06-19

## Goal

Move the orchestrator from "clients send requests and the proxy limits them" to
"clients submit task queues, the orchestrator understands the workload, chooses
context and parallel slots, prepares the right LM Studio load, and executes work
without clients managing GPU/model lifecycle."

The target behavior:

1. A client, such as `zotero-html-translate-worker`, submits a tenant-scoped HTML
   translation queue.
2. The orchestrator stores tasks durably with idempotency.
3. The orchestrator estimates or reads per-task input/output token budgets.
4. The orchestrator computes a per-model `context_plan`.
5. Lifecycle compares the plan with current backend state and GPU inventory.
6. Lifecycle reuses, drains, reloads, or starts backends as needed.
7. A task executor claims queued work, sends OpenAI-compatible calls through the
   selected backend, and writes final task status/result metadata.

## Current State

Implemented:

1. Strict `llmo.task.v1` documentation and examples.
2. `POST /tasks/queue` accepts a tenant-scoped batch of tasks.
3. In-memory task store tracks accepted tasks and tenant-scoped idempotency.
4. Queue proxy computes `queue_lengths` per model.
5. Queue proxy computes first-pass `context_plans`:
   - estimated input tokens;
   - output budget;
   - required context per task;
   - recommended `lms_context_length`;
   - recommended `lms_parallel`;
   - oversized tasks.
6. Queue proxy forwards `queue_lengths` and `context_plans` to lifecycle
   `/reconcile`.
7. Lifecycle can already start, warm, mark ready, drain, stop, and cleanup
   backend instances.
8. LM Studio can be controlled through `lms load` and `lms unload` when the CLI
   is available.

Important gap:

Lifecycle currently does not yet make placement/reload decisions from
`context_plans`, and tasks are not yet executed durably by the orchestrator.

## Design Principles

1. Client projects describe work, not backend mechanics.
2. `tenant` is the isolation boundary for task ids, idempotency, status lists,
   quotas, fairness, and metrics.
3. `project`, `service`, and `task` describe who inside the tenant produced the
   work.
4. Resource fields are requests and estimates; server-side policy wins.
5. Context planning uses buckets and hysteresis to avoid reload churn.
6. LM Studio reloads must be graceful: drain first, unload only owned loads, then
   reload and warm up.
7. Storage implementation is hidden behind a `TaskStore` interface.

## Phase 1: Durable Task Store

Replace the in-memory task store with a storage interface and a production
implementation.

### Tasks

1. Define `TaskStore` interface:
   - `submit_many(tasks)`;
   - `get_task(tenant, task_id)`;
   - `list_tasks(tenant, filters)`;
   - `update_state(task_id, state, metadata)`;
   - `claim_next(tenant/model/priority constraints)`;
   - `record_result(task_id, result/artifacts/usage)`;
   - `record_error(task_id, error)`;
   - `cancel_task(task_id)`;
   - `queue_lengths()`;
   - `context_plans()`.
2. Add `PostgresTaskStore`.
3. Keep `InMemoryTaskStore` only for tests/dev fallback.
4. Add migrations or startup schema creation for:
   - `tasks`;
   - `task_events`;
   - `task_results`;
   - idempotency index.
5. Enforce unique `(tenant, idempotency_key)`.
6. Add tenant-scoped list/status endpoints.

### Acceptance Criteria

1. Restarting queue proxy does not lose queued tasks.
2. Re-submitting the same tenant/idempotency key returns the original task.
3. One tenant cannot list or fetch another tenant's tasks.
4. Existing `/tasks/queue` response shape stays compatible.

## Phase 2: Context Plan Contract

Make context planning explicit and stable enough for lifecycle to consume.

### Tasks

1. Add a typed `ContextPlan` model shared by queue proxy and lifecycle.
2. Include:
   - `queued_tasks`;
   - `max_required_context_tokens`;
   - `recommended_lms_context_length`;
   - `requested_parallel`;
   - `recommended_lms_parallel`;
   - `total_slot_context_tokens`;
   - `context_cap_tokens`;
   - `oversized_tasks`;
   - `reload_required`;
   - `policy_reason`.
3. Add per-task token estimate sources:
   - caller-provided `tokens.estimated_input_tokens`;
   - payload text estimate;
   - future artifact pre-scan for HTML files.
4. Add configurable context buckets per model:
   - default: `4096`, `8192`, `16384`, `32768`, `65536`;
   - model-specific max context.
5. Add policy caps:
   - max parallel slots;
   - max context length;
   - max total slot context.

### Acceptance Criteria

1. A mixed queue of small and large HTML tasks produces a deterministic context
   plan.
2. Oversized tasks are rejected or marked blocked before model reload.
3. Context plan is included in `/tasks/queue`, `/tasks/status`, and lifecycle
   reconcile responses.

## Phase 3: Lifecycle Uses Context Plans

Teach lifecycle to choose backend shape from queue contents, not only queue
length.

### Tasks

1. Extend `/reconcile` input to accept `context_plans`.
2. Convert each `ContextPlan` into desired lifecycle overrides:
   - `lms_context_length`;
   - `lms_parallel`;
   - estimated VRAM;
   - queue timeout;
   - idle TTL.
3. Add `desired_backend_shape(profile, context_plan)`.
4. Update scheduler decisions to include:
   - `reuse`;
   - `start`;
   - `reload`;
   - `drain`;
   - `noop`;
   - `reject_oversized`.
5. Add tests for:
   - queue fits current load;
   - queue needs larger context;
   - queue needs fewer/more slots;
   - insufficient VRAM for planned shape.

### Acceptance Criteria

1. Lifecycle returns a clear `reload` decision when current LM Studio load cannot
   satisfy the context plan.
2. Lifecycle does not reload when current context/parallel already satisfy the
   queue.
3. Scheduler accounts for planned VRAM before choosing a GPU.

## Phase 4: LM Studio State Adapter

Add an adapter that can inspect and manage the current LM Studio load safely.

### Tasks

1. Parse `lms ps` into structured state:
   - identifier;
   - model;
   - status;
   - size;
   - context;
   - parallel;
   - device;
   - TTL.
2. Add `lms load --estimate-only` integration for planned shapes.
3. Record whether lifecycle owns a load:
   - loaded by lifecycle;
   - pre-existing external load;
   - CLI unavailable fallback.
4. Add adapter operations:
   - `inspect(model)`;
   - `estimate(model, context, parallel, gpu)`;
   - `load(model, context, parallel, gpu, ttl)`;
   - `unload(identifier)`;
   - `can_reload(instance, plan)`.
5. Store current LM Studio shape in backend registry metadata.

### Acceptance Criteria

1. Lifecycle can report actual current LM Studio `context` and `parallel`.
2. Lifecycle can detect mismatch with `context_plan`.
3. Lifecycle refuses to unload a pre-existing model unless policy explicitly
   allows taking ownership.

## Phase 5: Graceful Reload

Implement reload without interrupting active requests.

### Tasks

1. Add backend state transitions:
   - `ready -> draining -> stopping -> starting -> warming -> ready`;
   - failure path to `failed`.
2. Stop routing new tasks to draining backend.
3. Wait until `active_requests == 0`.
4. Unload only lifecycle-owned LM Studio loads.
5. Load planned context/parallel.
6. Warm up.
7. Mark ready.
8. Add reload hysteresis:
   - minimum dwell time;
   - context bucket thresholds;
   - do not shrink context during active batch unless idle;
   - avoid reload if benefit is below threshold.

### Acceptance Criteria

1. Active requests finish before unload.
2. New tasks wait or route elsewhere during reload.
3. Failed reload leaves a clear error and does not corrupt task state.
4. The system does not reload repeatedly while tasks are still arriving.

## Phase 6: Durable Task Executor

Move from "prepare capacity for external workers" to "orchestrator executes
durable tasks."

### Tasks

1. Add task worker loop in queue proxy or a dedicated service.
2. Claim tasks fairly by `(tenant, priority, model)`.
3. Build OpenAI-compatible payload from stored task payload and orchestration.
4. Route through existing queue proxy/backend resolver.
5. Store:
   - final response JSON or artifact refs;
   - token usage;
   - timing;
   - backend instance;
   - GPU ids;
   - error object.
6. Add:
   - `GET /tasks/{task_id}`;
   - `GET /tasks`;
   - `DELETE /tasks/{task_id}`.
7. Add retry policy:
   - retry transient upstream failures;
   - do not retry validation/token errors;
   - respect idempotency.

### Acceptance Criteria

1. `zotero-html-translate-worker` can submit a queue and poll status without
   holding long HTTP requests.
2. Completed tasks survive service restart.
3. Failed tasks expose stable error types.
4. Tenant isolation is enforced for all task APIs.

## Phase 7: Zotero HTML Translation Integration

Make the first real client use the protocol.

### Tasks

1. In `D:/Elvis_projects/Zotero_Automation/zotero-html-translate-worker`, add a
   queue submission mode.
2. Worker sends:
   - `tenant: elvis`;
   - `project: zotero`;
   - `service: zotero-html-translate-worker`;
   - `task: html_translate`;
   - artifact refs for `02.en.polish.html` and `03.ru.translate.html`;
   - per-task token estimates from HTML visible text.
3. Add a poller for durable task status once Phase 6 exists.
4. Keep old synchronous mode during migration.

### Acceptance Criteria

1. Worker can submit a batch without knowing GPU ids or LM Studio state.
2. Orchestrator chooses context/parallel from the submitted HTML sizes.
3. Worker receives task ids and status.

## Phase 8: Observability And Operations

Make the system explain itself.

### Tasks

1. Metrics:
   - tasks by tenant/project/service/task/state;
   - queue wait time;
   - execution time;
   - reload count;
   - reload failure count;
   - planned vs actual context;
   - planned vs actual VRAM.
2. Logs:
   - context plan decision;
   - scheduler decision;
   - reload decision;
   - task claim/finish/fail.
3. CLI:
   - `llmoctl tasks`;
   - `llmoctl task <id>`;
   - `llmoctl queue submit`;
   - `llmoctl explain-plan`.

### Acceptance Criteria

1. Operator can answer why a task is waiting.
2. Operator can see why a model was reloaded.
3. Operator can see which tenant/project is consuming slots.

## Configuration Changes

Recommended immediate config fix for `zotero-html-translate`:

```yaml
estimated_vram_gb: 26
safety_margin_gb: 2
lms_context_length: 32768
lms_parallel: 2
max_active_requests: 2
max_replicas: 1
```

Reason: local `lms load --estimate-only` reports about `25.46 GiB` for
`p6_google_gemma-4-26b-a4b@q6_k` at `context=32768`, while current config
reserves only `20 + 1 GiB`.

## Suggested Implementation Order

1. Durable `TaskStore`.
2. Shared typed `ContextPlan`.
3. Lifecycle reconcile consumes `context_plans`.
4. LM Studio inspect/estimate adapter.
5. Graceful reload policy.
6. Durable task executor.
7. Zotero worker queue submission.
8. Observability/CLI polish.
