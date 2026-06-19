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
3. `TaskStore` interface with in-memory and JSON-file implementations tracks
   accepted tasks and tenant-scoped idempotency.
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
9. Synchronous `/v1/...` requests that declare `schema_version: llmo.task.v1`
   are rejected when required identity fields, priority, or resource hints are
   malformed.
10. Queue proxy schedules a delayed reconcile after durable queues become empty,
    so lifecycle can unload owned idle LM Studio loads after the configured idle
    TTL.

Important gap:

The orchestrator has a working task-driven loop. Remaining production work is
mostly operational hardening outside the core loop: formal Postgres migrations,
real Postgres multi-worker integration tests, dashboards/structured logs, and
the first external worker migration to executable task payloads.

## Confirmed Product Decisions

1. Postgres is the target durable storage backend. JSON remains a local
   development and smoke-test backend.
2. Employers submit the actual work definition. If a prompt is needed, the
   employer must provide it as an OpenAI-compatible `payload` or explicit task
   template. The orchestrator must not invent task prompts.
3. Once a task is accepted, the orchestrator owns execution: backend selection,
   retry decisions, capacity reconciliation, lease handling, and final status.
4. For now all employers have equal scheduling priority. The `priority` field
   stays in the protocol for future policy, but current fairness must not let
   one employer group monopolize the queue.
5. Lifecycle must reconcile registry state with live LM Studio state.
6. Reload hysteresis is required before production use, so small context/slot
   changes do not unload and reload models repeatedly.
7. VRAM planning must compare current load, planned future load, and ownership:
   lifecycle may unload/reload only loads it owns, and must treat pre-existing
   external LM Studio loads as reserved capacity unless policy explicitly takes
   ownership.
8. Observability must explain waiting tasks, reload decisions, planned vs actual
   context/VRAM, and tenant/project/service usage.
9. Operator CLI commands are required for day-to-day inspection and control.

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

Status:

1. Done: `TaskStore` interface for queue submission, queue lengths, and context
   plans.
2. Done: `JsonFileTaskStore` selected by `TASK_STORE_PATH` for durable local
   queue state and restart-safe idempotency.
3. Done: in-memory store remains the default dev/test fallback.
4. Done: interface supports fetch/list/cancel/claim/result/error operations.
5. Done: `PostgresTaskStore` selected by `TASK_STORE_BACKEND=postgres` and
   `TASK_STORE_DSN`, with startup schema creation.
6. Done: retry metadata (`attempt_count`, `next_attempt_at`) is stored durably
   in JSON and Postgres.
7. Done: Postgres startup schema records `task_store_schema_version` and
   rejects a database schema that is newer than the running code.
8. Next: add migration tooling and integration tests against a real Postgres
   container for multi-worker production execution.

### Tasks

1. Extend `TaskStore` interface:
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

Status:

1. Done: `/plan` and `/reconcile` parse `context_plans`.
2. Done: lifecycle derives desired `lms_context_length` and `lms_parallel` from
   queue contents.
3. Done: start decisions include the planned LM Studio shape.
4. Done: ready LM Studio backends with too-small context/parallel return a
   `reload` decision instead of silently reusing the load.
5. Done: reload decisions are executed through the graceful reload behavior
   described in Phase 5.

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

Status:

1. Done: `lms ps --json` parser for loaded model identifier, model key,
   context, parallel, GPU, TTL, and raw metadata.
2. Done: `lms load --estimate-only` command builder and GPU-memory parser.
3. Done: lifecycle records planned LM Studio shape in backend registry metadata.
4. Done: dry-run lifecycle no longer calls real LM Studio load/unload commands.
5. Done: lifecycle compares reload decisions against live `lms ps --json`
   shape before relying on registry metadata.
6. Done: lifecycle can use `lms load --estimate-only` to override planned VRAM
   for the future backend shape.
7. Done: lifecycle persists live LM Studio reconciliation results in backend
   registry metadata.
8. Done: matching pre-existing LM Studio loads are represented as `external`
   backend records and their estimated VRAM is treated as reserved capacity.
9. Next: add takeover policy hooks only if an operator explicitly wants
   lifecycle to assume ownership of external loads.

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
6. Distinguish lifecycle-owned loads from pre-existing user/external loads.
7. Treat unowned loads as unavailable/reserved capacity unless takeover policy
   is explicitly enabled.

### Acceptance Criteria

1. Lifecycle can report actual current LM Studio `context` and `parallel`.
2. Lifecycle can detect mismatch with `context_plan`.
3. Lifecycle refuses to unload a pre-existing model unless policy explicitly
   allows taking ownership.

## Phase 5: Graceful Reload

Implement reload without interrupting active requests.

Status:

1. Done: lifecycle emits `reload` when a ready LM Studio backend is too small
   for the context plan.
2. Done: reconcile marks active mismatched backends as `draining` instead of
   unloading them.
3. Done: idle lifecycle-owned LM Studio backends reload through
   stop/start/warmup.
4. Done: pre-existing/unowned LM Studio loads are not unloaded.
5. Done: reload hysteresis avoids churn for bucket-only and non-critical shape
   changes.
6. Done: live LM Studio ownership reconciliation distinguishes lifecycle-owned
   loads from external loads before reload/cleanup.
7. Done: operator-facing plan explanations are exposed through lifecycle and
   `llmoctl explain-plan`.
8. Done: lifecycle exports reload and live LM Studio reconciliation counters.
9. Done: queue proxy re-reconciles capacity after task completion/failure/cancel
   and schedules a delayed empty-queue reconcile for idle unload.

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
9. Compare planned shape against current live LM Studio state before reload.

### Acceptance Criteria

1. Active requests finish before unload.
2. New tasks wait or route elsewhere during reload.
3. Failed reload leaves a clear error and does not corrupt task state.
4. The system does not reload repeatedly while tasks are still arriving.
5. Lifecycle does not reload only because the configured context bucket is
   larger when the current live context can still fit queued tasks.
6. Non-critical shape improvements, such as a larger parallel target, respect
   `reload_min_dwell_seconds` before reload.

## Phase 6: Durable Task Executor

Move from "prepare capacity for external workers" to "orchestrator executes
durable tasks."

Status:

1. Done: `TaskStore` supports tenant-scoped get/list/cancel, claim, result, and
   error recording.
2. Done: `GET /tasks`, `GET /tasks/{task_id}`, and `DELETE /tasks/{task_id}`
   expose tenant-scoped task status.
3. Done: optional in-process executor, enabled by `TASK_EXECUTOR_ENABLED`,
   claims queued tasks with stored OpenAI-compatible payloads.
4. Done: executor routes through backend resolver, records upstream results, and
   releases registry leases.
5. Done: executor-owned retry policy for transient backend errors.
6. Done: durable tasks expose `attempt_count`, `next_attempt_at`, and stable
   retryable/permanent error metadata.
7. Done: task claiming uses equal-priority fairness across
   `(tenant, project, service, task, priority, model)` groups, so one employer
   group cannot drain its whole batch before another due group gets a turn.
8. Done: first task execution metrics for events, errors, queue wait,
   execution duration, and current task state counts.
9. Done: executor validates endpoint-specific OpenAI-compatible payload shape
   before routing, so worker metadata is not accidentally sent to an LLM
   backend as a chat request.
10. Done: queue admission can render employer-provided `payload_template`
    objects with per-task `template_vars` into stored OpenAI-compatible payloads.
11. Next: update the Zotero worker to use `payload_template` or direct
    OpenAI-compatible payloads for durable execution.

### Tasks

1. Add task worker loop in queue proxy or a dedicated service.
2. Claim tasks fairly by `(tenant, project, service, task, priority, model)`.
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
8. Keep retry decisions inside the orchestrator. Employers submit work once and
   poll durable task status.

### Acceptance Criteria

1. `zotero-html-translate-worker` can submit a queue and poll status without
   holding long HTTP requests.
2. Completed tasks survive service restart.
3. Failed tasks expose stable error types.
4. Tenant isolation is enforced for all task APIs.
5. Retryable failures return to `queued` with `next_attempt_at`; permanent
   failures end in `failed` with `retryable: false`.
6. Equal-priority claiming rotates between employer groups instead of draining
   one group's entire queue first.

## Phase 7: Zotero HTML Translation Integration

Make the first real client use the protocol.

Status:

1. Observed: `zotero-html-translate-worker` can build and submit a task queue,
   but its current per-task `payload` is worker-runner metadata, not an
   executable OpenAI-compatible chat payload.
2. Done on orchestrator side: such a non-executable payload now fails with
   stable `invalid_task_payload` and `retryable: false` instead of being sent to
   LM Studio.
3. Next: update the worker to submit either a real OpenAI-compatible payload
   for each durable task or an explicit employer-owned template that the
   orchestrator can render without inventing prompts. The orchestrator side of
   `payload_template` rendering is implemented; the worker still needs to send
   text/template variables instead of only runner metadata.

### Tasks

1. In `D:/Elvis_projects/Zotero_Automation/zotero-html-translate-worker`, add a
   queue submission mode.
2. Worker sends:
   - `tenant: elvis`;
   - `project: zotero`;
   - `service: zotero-html-translate-worker`;
   - `task: html_translate`;
   - OpenAI-compatible payload or explicit prompt/template for each task;
   - artifact refs for `02.en.polish.html` and `03.ru.translate.html`;
   - per-task token estimates from HTML visible text.
3. Add a poller for durable task status once Phase 6 exists.
4. Keep old synchronous mode during migration.

### Acceptance Criteria

1. Worker can submit a batch without knowing GPU ids or LM Studio state.
2. Orchestrator chooses context/parallel from submitted token estimates and
   server-side policy.
3. Worker receives task ids and status.
4. Orchestrator does not generate the translation prompt unless the employer
   supplied it as task data.

## Phase 8: Observability And Operations

Make the system explain itself.

Status:

1. Done: durable task metrics are exported:
   - task lifecycle events by tenant/project/service/task/model;
   - task errors by stable error type and retryability;
   - current tasks by state;
   - queue wait duration;
   - execution duration.
2. Done: `llmoctl tasks`, `llmoctl task <id>`, and `llmoctl cancel-task <id>`
   inspect and control tenant-scoped durable tasks.
3. Done: lifecycle `POST /explain-plan`, queue proxy `GET /tasks/explain`, and
   `llmoctl explain-plan` explain current queue placement/reload decisions.
4. Done: lifecycle exports reload and live LM Studio reconciliation counters.
5. Next: add richer structured logs, dashboards, and deeper planned-vs-actual
   GPU/context metrics.

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
   - `llmoctl cancel-task <id>`;
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

## Historical Implementation Order

1. Durable `TaskStore`.
2. Shared typed `ContextPlan`.
3. Lifecycle reconcile consumes `context_plans`.
4. LM Studio inspect/estimate adapter.
5. Graceful reload policy.
6. Durable task executor.
7. Task retry policy and equal-priority task claiming.
8. Zotero worker queue submission.
9. Observability/CLI polish.
