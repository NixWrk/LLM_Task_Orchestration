# local-llm-orchestrator

Local OpenAI-compatible LLM orchestrator for servers with several GPUs and several services using local models.

The goal is to make one controlled entry point for all internal LLM traffic:

- Queue proxy is the public API endpoint for internal services.
- Queue proxy controls per-model concurrency, queue size, queue timeout, and token budget.
- Queue proxy can route through the lifecycle backend registry when ready backend instances exist.
- LiteLLM handles OpenAI-compatible routing and provider abstraction.
- LM Studio runs locally on the host and serves the model for the first backend.
- Postgres and Redis are available for LiteLLM state.
- Healthcheck verifies LM Studio and the full queue proxy -> LiteLLM -> backend path.
- GPU inventory exposes GPU/VRAM state for scheduling.
- Lifecycle service calculates dry-run model placement and backend registry state.
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
  -> GPU Inventory :4200
      -> nvidia-smi / fake inventory
  -> Lifecycle :4300
      -> scheduler
      -> backend registry
      -> Healthcheck service :8080
      -> Prometheus :9090
      -> Grafana :3000
```

LM Studio is not exposed by this compose file. It should run on the host and be reachable from Docker through `host.docker.internal`.

LiteLLM is still published on `:4000` for debugging, but internal services should use the queue proxy on `:4100`.

## Prerequisites

- Docker Desktop or Docker Engine with Compose.
- NVIDIA Container Toolkit / Docker GPU passthrough for real vLLM GPU serving.
- LM Studio installed.
- A local model downloaded in LM Studio.
- Optional: LM Studio CLI `lms`.

For a repeatable local preparation run that installs Python dependencies, pulls Docker images, builds services, runs tests, and verifies GPU passthrough, see [Real Run Preparation](docs/REAL_RUN_PREP.md).

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
- `lifecycle.estimated_vram_gb`: VRAM reservation for scheduler placement.
- `lifecycle.safety_margin_gb`: extra VRAM headroom.
- `lifecycle.preferred_gpus`: `auto` or explicit GPU ids such as `gpu0`.
- `lifecycle.min_replicas` / `lifecycle.max_replicas`: desired model replica bounds.

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
- GPU inventory: `http://localhost:4200/gpus`
- Lifecycle registry: `http://localhost:4300/registry`
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

## GPU Control Plane

The GPU management layer can run in either dry-run or real Docker mode. Dry-run mode is the default and records planned backend instances without starting containers. Real Docker mode is enabled with `LIFECYCLE_DRY_RUN=false` and uses the Docker vLLM adapter.

For the external contract other programs should use to request capacity, model startup, GPU constraints, and task-specific limits, see [Resource Request API](docs/RESOURCE_REQUEST_API.md).

For real vLLM container launching, Docker socket deployment, healthcheck/warmup, and idle stop behavior, see [Docker vLLM Runtime Adapter](docs/DOCKER_VLLM_RUNTIME.md).

GPU inventory:

```powershell
Invoke-RestMethod http://localhost:4200/gpus
```

Lifecycle placement plan:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:4300/plan `
  -ContentType "application/json" `
  -Body '{"queue_lengths":{"local-main":1}}'
```

Dry-run reconcile creates a registry entry for the planned backend instance:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:4300/reconcile `
  -ContentType "application/json" `
  -Body '{"queue_lengths":{"local-main":1}}'
```

On a machine without NVIDIA drivers, set `GPU_INVENTORY_FAKE_GPU_INVENTORY_JSON` in `.env`:

```json
{"gpus":[{"id":"gpu0","index":0,"name":"fake","memory_total_mb":24576,"memory_used_mb":2048}]}
```

Queue proxy can use ready HTTP backends from the lifecycle registry:

```text
ENABLE_BACKEND_REGISTRY_ROUTING=true
```

By default it falls back to `UPSTREAM_LITELLM_BASE_URL` when the registry has no ready HTTP backend. To force registry-only routing:

```text
REQUIRE_BACKEND_REGISTRY_BACKEND=true
```

When registry routing is enabled, queue proxy leases the selected backend before forwarding the request and releases it when the upstream response finishes. That keeps `active_requests` in the lifecycle registry current enough for least-active routing decisions.

The lifecycle service now has a runtime adapter layer. Dry-run mode records the command that would be used. For a vLLM model profile, the generated command is shaped like:

```powershell
docker run -d `
  --name llm-<instance> `
  --gpus device=0 `
  -p 8100:8000 `
  vllm/vllm-openai:latest `
  --model /models/qwen `
  --served-model-name qwen `
  --host 0.0.0.0 `
  --port 8000
```

Real Docker launching is intentionally opt-in:

```text
LIFECYCLE_DRY_RUN=false
```

For real container launching, the lifecycle image includes the Docker CLI and the compose file mounts `/var/run/docker.sock`. A production vLLM profile should include explicit model volumes and runtime settings:

```yaml
models:
  qwen-14b:
    public_name: qwen-14b
    backend_model: qwen-14b
    lifecycle:
      runtime: vllm
      artifact: D:/models/qwen-14b
      runtime_image: vllm/vllm-openai:latest
      host_port_start: 8100
      container_port: 8000
      public_host: host.docker.internal
      volumes:
        - host_path: D:/models/qwen-14b
          container_path: /models/qwen-14b
          mode: ro
      environment:
        HF_HOME: /root/.cache/huggingface
      runtime_extra_args:
        - --max-model-len
        - "8192"
      healthcheck_path: /v1/models
      startup_timeout_seconds: 120
      healthcheck_interval_seconds: 2
      warmup_enabled: true
      warmup_prompt: "Return exactly: ok"
      warmup_max_tokens: 8
      estimated_vram_gb: 16
      safety_margin_gb: 2
      min_replicas: 1
      max_replicas: 2
      idle_ttl_seconds: 900
      preferred_gpus: [gpu0, gpu1]
```

When `LIFECYCLE_DRY_RUN=false`, lifecycle starts the container, waits for `/v1/models`, sends a warmup chat completion, then marks the backend `ready`. Idle ready instances above `min_replicas` are marked `draining`, stopped with `docker stop`, and then marked `stopped`.

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

Integration tests start a fake OpenAI-compatible backend and a real queue proxy on temporary local ports:

```powershell
python -m pytest tests\integration
```

You can also start the fake backend through Compose for manual debugging:

```powershell
docker compose --profile test up -d --build fake-backend
```

## Current Scope

Implemented now:

- Phase 1 compose and LiteLLM configuration.
- Queue proxy for per-model concurrency, queueing, and token budget enforcement.
- Fake OpenAI-compatible backend for integration tests.
- Integration tests for non-streaming, streaming, token rejection, queue overflow, queue timeout, and upstream failure.
- GPU inventory service with `nvidia-smi` parser and fake inventory mode.
- Lifecycle dry-run scheduler with backend registry and VRAM-aware placement.
- Registry-aware queue proxy routing.
- Active request lease/release accounting between queue proxy and lifecycle registry.
- Lifecycle runtime adapter framework with Docker vLLM command generation.
- Production-oriented Docker vLLM lifecycle: model volumes, Docker socket/CLI launch, healthcheck, warmup, `starting -> ready`, and idle drain/stop.
- Real-run preparation script and documentation.
- Environment-driven settings.
- Smoke test scripts.
- Basic FastAPI healthcheck with Prometheus metrics.

Next phases:

- A concrete real model profile for the server's selected local model weights.
- Optional SGLang/LM Studio runtime adapters.
- Compatibility tests for streaming, Responses API, timeouts, and backend failures.
- Reverse proxy and TLS for controlled non-local access.
