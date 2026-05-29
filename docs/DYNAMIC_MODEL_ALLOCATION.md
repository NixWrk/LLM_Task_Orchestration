# Dynamic Model Allocation

Applications do not need to know which GPU or backend process is ready. They send an OpenAI-compatible request to queue proxy and name the model they want.

## Chat Request

```http
POST http://localhost:4100/v1/chat/completions
Content-Type: application/json
Authorization: Bearer <QUEUE_PROXY_API_KEY if configured>
```

```json
{
  "model": "qwen3_5_9b_q6k",
  "messages": [
    {"role": "user", "content": "Return exactly: ok"}
  ],
  "max_tokens": 64,
  "orchestration": {
    "gpu": "auto",
    "max_parallel": 1,
    "max_queued_requests": 8,
    "queue_timeout_seconds": 30,
    "estimated_vram_gb": 8,
    "safety_margin_gb": 1,
    "idle_ttl_seconds": 900,
    "tokens": {
      "max_output_tokens": 512,
      "max_total_tokens": 8192
    }
  }
}
```

`orchestration` is consumed by queue proxy and lifecycle. It is removed before forwarding to LM Studio, vLLM, or another OpenAI-compatible backend.

The request flow is:

1. Queue proxy resolves an explicit model policy, or builds a dynamic policy from `defaults`.
2. Queue proxy applies request-level limits from `orchestration`, capped by the configured policy.
3. Queue proxy asks lifecycle for a ready backend when registry routing is enabled.
4. Lifecycle checks existing ready instances.
5. If none exists, lifecycle creates an allocation from `dynamic_models`.
6. For `runtime: lmstudio`, lifecycle verifies `/v1/models`, warms the requested model, and marks the backend `ready`.
7. Queue proxy leases the backend, forwards the request, streams the response, and releases the lease.

## Allocation API

Queue proxy calls this endpoint automatically, but other internal controllers can call it directly:

```http
POST http://localhost:4300/allocations
Content-Type: application/json
```

```json
{
  "model": "qwen3_5_9b_q6k",
  "orchestration": {
    "runtime": "lmstudio",
    "base_url": "http://host.docker.internal:1234/v1",
    "gpu": "gpu0",
    "estimated_vram_gb": 8,
    "safety_margin_gb": 1,
    "idle_ttl_seconds": 900
  }
}
```

Success response:

```json
{
  "model": "qwen3_5_9b_q6k",
  "created": true,
  "decision": {
    "model": "qwen3_5_9b_q6k",
    "action": "start",
    "gpu_id": "gpu0",
    "reason": "vram_available",
    "required_vram_mb": 9216,
    "available_vram_mb": 49000
  },
  "instance": {
    "model": "qwen3_5_9b_q6k",
    "backend_model": "qwen3_5_9b_q6k",
    "runtime": "lmstudio",
    "base_url": "http://host.docker.internal:1234/v1",
    "state": "ready"
  }
}
```

If no GPU has enough free VRAM, lifecycle returns `409` with the placement decision and no instance.

## Required Settings

Dynamic allocation is enabled by:

```yaml
dynamic_models:
  enabled: true
  source: lmstudio
  lifecycle:
    runtime: lmstudio
    base_url: http://host.docker.internal:1234/v1
    estimated_vram_gb: 8
    safety_margin_gb: 1
    min_replicas: 0
    max_replicas: 1
    idle_ttl_seconds: 900
    preferred_gpus:
      - auto
```

Queue proxy should route through lifecycle:

```text
ENABLE_BACKEND_REGISTRY_ROUTING=true
```

Use strict mode when every request must have a lifecycle-ready backend:

```text
REQUIRE_BACKEND_REGISTRY_BACKEND=true
```

With strict mode disabled, queue proxy may fall back to LiteLLM if allocation fails.
