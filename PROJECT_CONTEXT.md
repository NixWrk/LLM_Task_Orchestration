# Project Context

## Project Name

`local-llm-orchestrator`

## Purpose

This repository implements a local/self-hosted LLM orchestrator for a server with several GPUs and several internal services using LLMs.

The orchestrator must regulate:

- how many model instances are running;
- which GPU or backend owns each model instance;
- how many requests may run in parallel per model;
- how many requests may wait in queue per model;
- how long queued requests may wait;
- how many input/output/total tokens each request may use.

The initial implementation is based on:

- `Queue Proxy` as the public OpenAI-compatible entry point for internal services.
- `LiteLLM Proxy` as the public OpenAI-compatible gateway.
- `LM Studio` as the local model runtime.
- Small supporting services for health checks, lifecycle control, backpressure, observability, and safe deployment.

The target use cases are local development, small-team internal usage, and controlled self-hosted inference. The orchestrator is not an inference runtime by itself. It controls and routes traffic to runtimes such as LM Studio, vLLM, or SGLang.

## Target Architecture

```text
Client / OpenAI SDK / Codex-compatible client
  -> Reverse proxy
      -> Queue Proxy
          -> token budget policy
          -> per-model queue
          -> per-model concurrency limiter
          -> LiteLLM Proxy
              -> LM Studio / vLLM / SGLang backend
                  -> Local model on one or more GPUs
              -> Postgres
              -> Redis
      -> Healthcheck service
      -> Model lifecycle controller
      -> Metrics/logging stack
```

Minimal local architecture:

```text
Client
  -> Queue Proxy
      -> LiteLLM Proxy
          -> LM Studio
              -> Local model
```

## Main Components

### Queue Proxy

Queue Proxy is the default public entry point for internal services. It exposes OpenAI-compatible `/v1/...` routes and forwards accepted requests to LiteLLM.

Responsibilities:

- resolve request model to a model policy;
- enforce input/output/total token budgets;
- set default output token limits;
- clamp or reject oversized output requests according to policy;
- limit active requests per model;
- limit queued requests per model;
- return `429` for queue overflow or queue timeout;
- expose queue and active request metrics.

### LiteLLM Proxy

LiteLLM provides provider abstraction behind the queue proxy. It handles OpenAI-compatible routing, API key based access, model aliases, retry/timeout settings, optional virtual keys, and database-backed spend/state features.

The proxy routes requests to LM Studio as an OpenAI-compatible backend.

### LM Studio

LM Studio serves the actual model through OpenAI-compatible endpoints.

Expected local base URL:

```text
http://localhost:1234/v1
```

Required endpoints:

```text
GET  /v1/models
POST /v1/chat/completions
POST /v1/responses
POST /v1/embeddings
POST /v1/completions
```

### Postgres

Postgres is used for LiteLLM database-backed features such as virtual keys, users, budgets, spend tracking, and persistent gateway state.

### Redis

Redis is available for distributed routing/rate-limit state when multiple LiteLLM instances or multiple backends are introduced.

### Model Lifecycle Controller

The lifecycle controller is the future control-plane component that starts, stops, warms, and unloads model runtimes.

Responsibilities:

- keep `min_replicas` loaded for important models;
- scale up toward `max_replicas` when queue pressure is high;
- scale down idle models after TTL;
- place model instances on allowed GPUs;
- avoid starting models when estimated VRAM is unavailable;
- expose desired and actual model state.

### Healthcheck Service

The healthcheck service verifies:

1. LM Studio `/v1/models` responds.
2. The expected model identifier is visible.
3. A short completion through Queue Proxy -> LiteLLM succeeds.
4. Health state and latency are exported through Prometheus metrics.

## Implementation Priorities

### Phase 1: Minimal Working Orchestrated Gateway

1. Create `docker-compose.yml`.
2. Add LiteLLM config.
3. Connect LiteLLM to external LM Studio.
4. Add queue proxy.
5. Add per-model queue/concurrency/token policy.
6. Add `.env.example`.
7. Add smoke test scripts.
8. Document startup.

### Phase 2: Reliability

1. Add richer health checks.
2. Add lifecycle controller.
3. Add model warmup.
4. Add backend readiness checks.
5. Add integration tests.

### Phase 3: Backpressure

1. Persist queue and state in Redis if multiple queue proxy replicas are needed.
2. Add priority classes.
3. Add per-service quotas.
4. Add richer token accounting with tokenizer-specific estimators.
5. Add streaming compatibility tests.

### Phase 4: Observability

1. Add Prometheus.
2. Add Grafana dashboards.
3. Add structured logs.
4. Add GPU/RAM metrics.
5. Add latency and token metrics.

### Phase 5: Hardening

1. Add reverse proxy.
2. Add TLS.
3. Add IP allowlist or external auth.
4. Keep prompt logging disabled by default.
5. Add backup policy for Postgres.
6. Add failure-mode tests.

## Definition of Done for MVP

The MVP is complete when:

1. A client can send an OpenAI-compatible request to Queue Proxy.
2. Queue Proxy applies token and queue policy.
3. Queue Proxy forwards the request to LiteLLM.
4. LiteLLM forwards the request to LM Studio.
5. LM Studio returns a valid response.
6. A smoke test passes from the command line.
7. Configuration is environment-driven.
8. Basic healthcheck endpoint exists.
9. README contains startup instructions.
10. No secrets are committed.
11. LM Studio is not publicly exposed by compose.
12. Failure of LM Studio is reported clearly.
