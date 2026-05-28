# local-llm-orchestrator

Local OpenAI-compatible LLM orchestrator for servers with several GPUs and several services using local models.

The goal is to make one controlled entry point for all internal LLM traffic:

- Queue proxy is the public API endpoint for internal services.
- Queue proxy controls per-model concurrency, queue size, queue timeout, and token budget.
- LiteLLM handles OpenAI-compatible routing and provider abstraction.
- LM Studio runs locally on the host and serves the model for the first backend.
- Postgres and Redis are available for LiteLLM state.
- Healthcheck verifies LM Studio and the full queue proxy -> LiteLLM -> backend path.
- Prometheus and Grafana are wired for service metrics.

The first backend is LM Studio because it is convenient locally. For heavier multi-GPU serving, the intended migration path is to keep this orchestrator and replace or extend the backend with vLLM/SGLang instances.

## Architecture

```text
Service A / Service B / OpenAI SDK compatible client
  -> Queue Proxy :4100
      -> per-model token budget
      -> per-model queue
      -> per-model active request limiter
      -> LiteLLM Proxy :4000
          -> LM Studio OpenAI-compatible API on the host :1234
          -> Postgres
          -> Redis
      -> Healthcheck service :8080
      -> Prometheus :9090
      -> Grafana :3000
```

LM Studio is not exposed by this compose file. It should run on the host and be reachable from Docker through `host.docker.internal`.

LiteLLM is still published on `:4000` for debugging, but internal services should use the queue proxy on `:4100`.

## Prerequisites

- Docker Desktop or Docker Engine with Compose.
- LM Studio installed.
- A local model downloaded in LM Studio.
- Optional: LM Studio CLI `lms`.

## Configure LM Studio

Start the LM Studio server on port `1234`.

With the desktop app, enable the local server from the Developer/API panel.

With the CLI:

```powershell
lms server start --port 1234
lms ls
lms load <model-key> --identifier local-main
```

Use the model identifier you load as `LMSTUDIO_MODEL_ID`.

## Configure This Orchestrator

Create a local environment file:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set at least:

```text
LMSTUDIO_MODEL_ID=local-main
LITELLM_MODEL=openai/local-main
LITELLM_MASTER_KEY=sk-change-this-local-key
```

Model orchestration policy lives in:

```text
config/orchestrator.yaml
```

For each public model you can set:

- `max_active_requests`: how many requests may run at once.
- `max_queued_requests`: how many requests may wait.
- `queue_timeout_seconds`: how long a request may wait for a slot.
- `default_max_output_tokens`: output budget when the caller does not specify one.
- `max_input_tokens`: maximum estimated input size.
- `max_output_tokens`: maximum output budget.
- `max_total_tokens`: input estimate plus output budget.
- `lifecycle.min_replicas` / `lifecycle.max_replicas`: desired future model replica bounds.

For Docker Desktop on Windows and macOS, the default backend URL usually works:

```text
LMSTUDIO_OPENAI_BASE_URL=http://host.docker.internal:1234/v1
```

On Linux, keep the compose `extra_hosts` entry or set a host address such as:

```text
LMSTUDIO_OPENAI_BASE_URL=http://172.17.0.1:1234/v1
```

## Start

```powershell
docker compose up -d --build
```

Useful URLs:

- Queue proxy: `http://localhost:4100`
- LiteLLM debug endpoint: `http://localhost:4000`
- Healthcheck: `http://localhost:8080/ready`
- Metrics: `http://localhost:8080/metrics`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

## Smoke Test

PowerShell:

```powershell
.\scripts\smoke_test.ps1
```

Bash:

```bash
./scripts/smoke_test.sh
```

Expected result: HTTP 200 from LiteLLM and a valid OpenAI-compatible chat completion.

The smoke test goes through the queue proxy by default.

## Runtime Behavior

For a request to `/v1/chat/completions`, `/v1/responses`, `/v1/completions`, or `/v1/embeddings`, the queue proxy:

1. Reads the requested `model`.
2. Resolves model policy from `config/orchestrator.yaml`.
3. Estimates input tokens with a configurable chars-per-token heuristic.
4. Sets a default output token limit if the caller omitted one.
5. Clamps oversized output token requests unless the policy says to reject.
6. Rejects too-large input or total token budget with `413`.
7. Admits the request into the per-model queue.
8. Rejects queue overflow or queue timeout with `429`.
9. Forwards the request to LiteLLM.

This gives immediate protection when several internal services call the same local model at the same time.

## Development

Run the healthcheck service locally:

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -e ".[dev]"
.\.venv\Scripts\uvicorn app.main:app --app-dir services/healthcheck --reload --port 8080
```

Run tests:

```powershell
python -m pytest
```

## Current Scope

Implemented now:

- Phase 1 compose and LiteLLM configuration.
- Queue proxy for per-model concurrency, queueing, and token budget enforcement.
- Environment-driven settings.
- Smoke test scripts.
- Basic FastAPI healthcheck with Prometheus metrics.

Next phases:

- Lifecycle controller for model loading, warmup, idle unload, and GPU placement.
- Backend registry for several model runtimes across several GPUs.
- Compatibility tests for streaming, Responses API, timeouts, and backend failures.
- Reverse proxy and TLS for controlled non-local access.
