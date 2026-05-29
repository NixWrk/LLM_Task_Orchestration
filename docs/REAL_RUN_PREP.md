# Real Run Preparation

This project can be prepared for real local runs with one PowerShell command:

```powershell
.\scripts\prepare_real_run.ps1
```

If local PowerShell policy blocks script execution, run it for this process only:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\prepare_real_run.ps1
```

The script is idempotent and performs the same setup that was used on this machine.

## What It Prepares

- Creates `.env` from `.env.example` when `.env` is missing.
- Creates `.venv` and installs `.[dev]`.
- Runs the Python test suite.
- Pulls runtime images:
  - `postgres:16-alpine`
  - `redis:7-alpine`
  - `docker.litellm.ai/berriai/litellm:main-latest`
  - `prom/prometheus:v2.54.1`
  - `grafana/grafana-oss:11.2.0`
  - `nvidia/cuda:12.8.0-base-ubuntu22.04`
  - `vllm/vllm-openai:latest`
- Builds local Compose services.
- Builds the fake backend test image.
- Verifies Docker GPU passthrough with `nvidia-smi` inside a CUDA container.

Use these switches when you need a faster or partial run:

```powershell
.\scripts\prepare_real_run.ps1 -SkipDockerPull
.\scripts\prepare_real_run.ps1 -SkipVllmPull
.\scripts\prepare_real_run.ps1 -SkipGpuCheck
.\scripts\prepare_real_run.ps1 -SkipTests
```

## Verified On This Host

The host has Docker Desktop with NVIDIA GPU passthrough working. `nvidia-smi` was verified both on the host and from a Docker CUDA container. Two GPUs were visible:

- `NVIDIA RTX 6000 Ada Generation`, about 48 GB VRAM.
- `NVIDIA GeForce RTX 4080`, about 16 GB VRAM.

The following images were pulled or built successfully:

- `docker.litellm.ai/berriai/litellm:main-latest`
- `vllm/vllm-openai:latest`
- `nvidia/cuda:12.8.0-base-ubuntu22.04`
- Local images for `queue-proxy`, `gpu-inventory`, `lifecycle`, `healthcheck`, and `fake-backend`.

## What Is Still Local To Your Model

The script does not download model weights. The orchestrator needs an explicit local model path because the correct artifact depends on the model you want to serve, storage layout, license, and whether you use Hugging Face cache or a manually downloaded directory.

If the model is already downloaded in LM Studio, you usually do not need a vLLM model path. Start LM Studio, discover the model id with [LM Studio Models](LM_STUDIO_MODELS.md), and use `runtime: lmstudio` with `base_url: http://host.docker.internal:1234/v1`.

For a real vLLM backend, update `config/orchestrator.yaml`:

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
      docker_extra_args:
        - --ipc=host
      runtime_extra_args:
        - --max-model-len
        - "8192"
      estimated_vram_gb: 16
      safety_margin_gb: 2
      min_replicas: 1
      max_replicas: 2
      idle_ttl_seconds: 900
      preferred_gpus:
        - gpu0
        - gpu1
```

Then set:

```text
LIFECYCLE_DRY_RUN=false
ENABLE_BACKEND_REGISTRY_ROUTING=true
```

## Real Launch Check

After configuring a real model profile:

```powershell
docker compose up -d --build gpu-inventory lifecycle queue-proxy

Invoke-RestMethod http://localhost:4200/gpus

Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:4300/reconcile `
  -ContentType "application/json" `
  -Body '{"queue_lengths":{"qwen-14b":1}}'

Invoke-RestMethod http://localhost:4300/registry
```

The expected lifecycle path for a real vLLM instance is:

```text
starting -> ready
ready -> draining -> stopped
```

`starting -> ready` requires a successful `/v1/models` healthcheck and warmup request. Idle stop only applies to ready instances with `active_requests == 0` above `min_replicas`.
