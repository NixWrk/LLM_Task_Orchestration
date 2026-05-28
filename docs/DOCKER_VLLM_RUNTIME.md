# Docker vLLM Runtime Adapter

This document describes how lifecycle starts and stops real vLLM containers.

By default lifecycle runs in dry-run mode:

```text
LIFECYCLE_DRY_RUN=true
```

In dry-run mode, lifecycle records the Docker command that would be used but does not start containers.

## Enable Real Launching

Set:

```text
LIFECYCLE_DRY_RUN=false
LIFECYCLE_DOCKER_BINARY=docker
```

The lifecycle image includes the Docker CLI. The compose file mounts:

```yaml
- /var/run/docker.sock:/var/run/docker.sock
```

This allows lifecycle to run containers on the host Docker engine.

Security note: mounting the Docker socket gives the lifecycle container host-level Docker control. Only expose lifecycle on a trusted internal network.

## Model Profile

Example vLLM profile:

```yaml
models:
  qwen-14b:
    public_name: qwen-14b
    backend_model: qwen-14b
    aliases:
      - qwen-14b
    max_active_requests: 2
    max_queued_requests: 64
    queue_timeout_seconds: 30
    max_input_tokens: 8192
    max_output_tokens: 2048
    max_total_tokens: 10240
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
      docker_extra_args:
        - --ipc=host
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
      preferred_gpus:
        - gpu0
        - gpu1
```

Lifecycle maps `artifact` to the container path if it is inside one of the configured `volumes`. With the example above, the generated vLLM command uses:

```text
--model /models/qwen-14b
```

## State Transitions

When lifecycle starts a real vLLM backend:

```text
starting
  -> healthcheck /v1/models succeeds
  -> warming
  -> warmup chat completion succeeds
  -> ready
```

If healthcheck or warmup fails:

```text
starting/warming -> failed
```

When an instance is idle and above `min_replicas`:

```text
ready
  -> active_requests == 0
  -> idle_ttl_seconds elapsed
  -> draining
  -> docker stop <container>
  -> stopped
```

## Manual Test

Use fake GPU inventory on a machine without NVIDIA drivers:

```powershell
$env:GPU_INVENTORY_FAKE_GPU_INVENTORY_JSON='{"gpus":[{"id":"gpu0","index":0,"name":"fake","memory_total_mb":24576,"memory_used_mb":1024}]}'
```

Dry-run reconcile:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:4300/reconcile `
  -ContentType "application/json" `
  -Body '{"queue_lengths":{"qwen-14b":1}}'
```

Inspect registry:

```powershell
Invoke-RestMethod http://localhost:4300/registry
```

Only disable dry-run once model paths, Docker socket, GPU access, and vLLM image pull are known to work.
