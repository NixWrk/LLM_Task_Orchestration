# LM Studio Models

LM Studio is the fastest way to use the models that are already downloaded on this host. The orchestrator treats LM Studio as a private OpenAI-compatible backend.

## Start LM Studio

Start the local server on port `1234` from the LM Studio app, or with the CLI:

```powershell
lms server start --port 1234
```

Check status:

```powershell
lms server status
```

If the CLI prints `Timed out waiting for LM Studio daemon to start`, open the LM Studio desktop app once and enable the local server from the Developer/API panel. After the app is running, repeat `lms server status` or the discovery command.

The Docker services reach the host LM Studio server through:

```text
http://host.docker.internal:1234/v1
```

From the host itself, the equivalent URL is:

```text
http://localhost:1234/v1
```

## Discover Downloaded Models

Use the discovery script:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\discover_lmstudio_models.ps1
```

It tries three sources:

- LM Studio OpenAI-compatible `/v1/models`.
- `lms ls --json`.
- Common LM Studio model folders such as `%USERPROFILE%\.lmstudio\models`.

To save a machine-readable inventory:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\discover_lmstudio_models.ps1 `
  -OutputPath .\data\lmstudio-models.json
```

If LM Studio stores models outside the default folders:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\discover_lmstudio_models.ps1 `
  -ModelRoots "D:\LMStudio\Models","E:\Models"
```

## Configure One LM Studio Model

Set `.env` to the model identifier you load in LM Studio:

```text
LMSTUDIO_MODEL_ID=<lmstudio-model-id>
LITELLM_MODEL=openai/<lmstudio-model-id>
PUBLIC_MODEL_NAME=local-main
```

Keep `config/litellm.config.yaml` as the LiteLLM bridge to LM Studio. Then set the public model in `config/orchestrator.yaml`:

```yaml
models:
  local-main:
    public_name: local-main
    backend_model: <lmstudio-model-id>
    aliases:
      - local-main
      - <lmstudio-model-id>
    max_active_requests: 1
    max_queued_requests: 16
    queue_timeout_seconds: 30
    lifecycle:
      runtime: lmstudio
      base_url: http://host.docker.internal:1234/v1
      estimated_vram_gb: 8
      safety_margin_gb: 1
      min_replicas: 1
      max_replicas: 1
      idle_ttl_seconds: 3600
      preferred_gpus:
        - auto
```

`runtime: lmstudio` does not start a container. Lifecycle registers the already running LM Studio server, waits for `/v1/models`, sends a warmup request, then marks the backend `ready`.

To let lifecycle load and unload this model through the LM Studio CLI, add:

```yaml
lifecycle:
  runtime: lmstudio
  base_url: http://host.docker.internal:1234/v1
  load_strategy: cli-if-available
  lms_binary: lms
  lms_gpu: max
  lms_context_length: 8192
  lms_parallel: 1
  lms_ttl_seconds: 900
```

Lifecycle then runs `lms load <backend_model> --identifier <backend_model> --yes` before healthcheck/warmup and `lms unload <backend_model>` during idle stop, but only for models it actually loaded. If the identifier already exists, lifecycle treats it as pre-existing and leaves unloading to LM Studio/user TTL.

On Windows with Docker Compose, the lifecycle container normally cannot execute the host `lms.exe`. Use `cli-if-available` for Docker safety, or run lifecycle directly on the host when you want real CLI control.

## Verify GPU Loading

When LM Studio is the backend, model weights are loaded by `LM Studio.exe` on the
host, not by the Docker containers. It is normal for orchestrator containers,
LiteLLM, Postgres, Redis, Prometheus, or Grafana to use system RAM. That RAM use
does not prove the model is CPU-bound.

For GPU-backed loading, use an explicit CLI load command:

```powershell
lms load p6_google_gemma-4-26b-a4b@q6_k `
  --gpu max `
  --context-length 32768 `
  --parallel 1 `
  --ttl 3600 `
  --identifier p6_google_gemma-4-26b-a4b@q6_k `
  --yes
```

Check the resource estimate without loading:

```powershell
lms load p6_google_gemma-4-26b-a4b@q6_k `
  --gpu max `
  --context-length 32768 `
  --parallel 1 `
  --estimate-only `
  --yes
```

Expected signs of a GPU load:

1. `lms ps` shows the model with the requested context and parallel count.
2. `nvidia-smi` shows `LM Studio.exe` as a GPU process.
3. VRAM usage rises by roughly the model estimate.
4. A queue-proxy request to `http://localhost:4100/v1/chat/completions` succeeds
   for the same model identifier.

For the Zotero translation profile, the current baseline is:

```text
model: p6_google_gemma-4-26b-a4b@q6_k
context_length: 32768
parallel: 1
gpu: max
estimated_gpu_memory: about 25.5 GiB
```

If system RAM rises but `nvidia-smi` does not show `LM Studio.exe`, the model was
not GPU-offloaded as expected or the backend is not LM Studio. First verify the
actual loaded model with `lms ps`, then reload with `--gpu max`.

## Route Through The Registry

For direct routing to ready LM Studio/vLLM backends:

```text
ENABLE_BACKEND_REGISTRY_ROUTING=true
```

With fallback enabled, queue proxy uses LiteLLM when lifecycle has no ready backend. To require lifecycle readiness:

```text
REQUIRE_BACKEND_REGISTRY_BACKEND=true
```

## Multiple Downloaded Models

Add one `models.<name>` block per public model in `config/orchestrator.yaml`. Use conservative limits per model:

- Smaller 7B/8B quantized GGUF models: start with `estimated_vram_gb: 6` to `10`.
- Medium 14B quantized GGUF models: start with `estimated_vram_gb: 12` to `18`.
- Larger models: reserve enough VRAM for context, KV cache, and batching.

LM Studio itself decides how the model is loaded. The orchestrator controls admission, queues, token budgets, health/warmup, and routing. For hard multi-replica GPU placement, use the Docker vLLM adapter described in [Docker vLLM Runtime Adapter](DOCKER_VLLM_RUNTIME.md).

For dynamic models, `dynamic_models.auto_vram_from_lms: true` makes lifecycle estimate VRAM from `lms ls --json` fields such as `sizeBytes` and `maxContextLength`; it also stores compact LM Studio metadata like quantization and selected variant in the backend registry. Manual `estimated_vram_gb` remains available as a per-request override.

For application-side request examples where the app names the model and asks for GPU/token/concurrency constraints, see [Dynamic Model Allocation](DYNAMIC_MODEL_ALLOCATION.md).
