# Architecture Notes

This repository is organized around one OpenAI-compatible entry point and a separate GPU/model lifecycle control plane.

## Queue Proxy

`services/queue_proxy/queue_proxy/main.py` owns FastAPI routes and metrics wiring only.

Supporting modules:

- `request_preparation.py` parses OpenAI-compatible JSON requests, resolves model policy, applies token limits, and extracts orchestration metadata.
- `routing.py` selects a ready backend from lifecycle registry, asks lifecycle to allocate a dynamic backend when needed, leases the backend, and releases it after streaming completes.
- `forwarder.py` performs raw upstream forwarding and streaming response cleanup.
- `http_proxy.py` contains low-level URL/header normalization helpers.
- `policy.py` contains model policy and token budget rules.

## Lifecycle

`services/lifecycle/lifecycle/controller.py` is now a coordinator. It keeps the public control-plane methods and delegates runtime-specific work.

Supporting modules:

- `allocation_service.py` implements `POST /allocations`: dynamic policy checks, LM Studio metadata enrichment, GPU placement, instance creation, and initialization.
- `runtime.py` starts/stops/warms model backends through runtime adapters.
- `cleanup.py` drains idle backends and purges stale registry records.
- `registry.py` stores backend state in memory and delegates persistence to `registry_store.py`.
- `registry_store.py` writes the JSON registry through an atomic temp-file replace.
- `config.py` converts typed YAML config from `orchestrator_core.config` into lifecycle model profiles.

## Shared Packages

- `orchestrator_core` contains helpers shared by services: JSON logging, OpenAI URL normalization, Prometheus label escaping, and typed YAML loading.
- `orchestrator_client` is the HTTP client used by `llmoctl` and intended to be reused by a future MCP adapter.

## Test Layout

Unit tests stay next to their service when they cover service internals. Cross-service HTTP behavior lives under `tests/integration`.

Integration process helpers are in `tests/integration/support.py`, so scenario files can stay focused on queueing, streaming, dynamic allocation, denied models, embeddings, and idle cleanup behavior.
