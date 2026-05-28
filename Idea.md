\# PROJECT\_CONTEXT.md



\## 1. Project name



`local-llm-gateway`



\## 2. Purpose



This project implements a local/self-hosted LLM gateway based on:



\* `LiteLLM Proxy` as the public OpenAI-compatible gateway.

\* `LM Studio` as the local model runtime.

\* Additional infrastructure for health checks, model lifecycle control, request backpressure, observability, and safe deployment.



The system is intended for local development, small-team internal usage, and controlled self-hosted inference. It is not intended to replace high-throughput GPU runtimes such as `vLLM` or `SGLang`.



\## 3. Target architecture



```text

Client / OpenAI SDK / Codex-compatible client

&#x20; -> Reverse proxy

&#x20;     -> LiteLLM Proxy

&#x20;         -> LM Studio OpenAI-compatible server

&#x20;             -> Local model

&#x20;         -> Postgres

&#x20;         -> Redis

&#x20;     -> Healthcheck service

&#x20;     -> Model lifecycle controller

&#x20;     -> Metrics/logging stack

```



Minimal local architecture:



```text

Client

&#x20; -> LiteLLM Proxy

&#x20;     -> LM Studio

&#x20;         -> Local model

```



\## 4. Main components



\### 4.1. LiteLLM Proxy



LiteLLM Proxy is the public gateway. It provides:



\* OpenAI-compatible API surface.

\* API key based access.

\* Model aliases.

\* Routing configuration.

\* Retry and timeout settings.

\* Optional virtual keys and spend tracking with database-backed mode.



LiteLLM must route requests to LM Studio as to an OpenAI-compatible endpoint.



Expected LiteLLM model configuration:



```yaml

model\_list:

&#x20; - model\_name: local-main

&#x20;   litellm\_params:

&#x20;     model: openai/<lmstudio-model-id>

&#x20;     api\_base: http://lmstudio:1234/v1

&#x20;     api\_key: lm-studio

&#x20;     timeout: 120



general\_settings:

&#x20; master\_key: ${LITELLM\_MASTER\_KEY}

```



Rules:



\* `model\_name` is the public model name used by clients.

\* `litellm\_params.model` must use the `openai/` prefix for OpenAI-compatible chat completions.

\* `api\_base` must point to the LM Studio `/v1` base URL.

\* `api\_key` can be a dummy value unless LM Studio authentication is enabled.



\### 4.2. LM Studio



LM Studio is the local model runtime. It serves the actual model through an OpenAI-compatible API.



Expected base URL:



```text

http://lmstudio:1234/v1

```



Required endpoints:



```text

GET  /v1/models

POST /v1/chat/completions

POST /v1/responses

POST /v1/embeddings

POST /v1/completions

```



LM Studio may be run in one of two modes:



1\. Desktop app with server enabled.

2\. Headless service via `llmster`.



Preferred production-like local mode:



```bash

lms daemon up

```



\### 4.3. Postgres



Postgres is required when using database-backed LiteLLM features:



\* virtual keys;

\* users;

\* teams;

\* budgets;

\* spend tracking;

\* persistent gateway state.



Environment variable:



```bash

DATABASE\_URL=postgresql://litellm:litellm@postgres:5432/litellm

```



Postgres is optional for a minimal prototype.



\### 4.4. Redis



Redis is required for distributed or multi-instance gateway state.



Use Redis when:



\* more than one LiteLLM Proxy instance is running;

\* there are several LM Studio backends;

\* routing state must be shared;

\* cooldown/rate-limit state must survive process boundaries.



Redis is optional for a single-process local prototype.



\### 4.5. Reverse proxy



A reverse proxy should sit in front of LiteLLM for non-local access.



Recommended options:



\* Caddy;

\* Nginx;

\* Traefik.



Responsibilities:



\* TLS termination;

\* request size limits;

\* access logs;

\* IP allowlist if needed;

\* optional external authentication;

\* no direct public exposure of LM Studio.



LM Studio must remain private and accessible only from the internal network.



\## 5. Features to implement in this repository



\### 5.1. Healthcheck service



Implement a small service that checks whether LM Studio and the configured model are usable.



Required checks:



1\. `GET /v1/models` returns successfully.

2\. Expected model identifier exists or can be loaded by LM Studio.

3\. A short test completion succeeds.

4\. Latency and error rate are recorded.

5\. Health state is exported for monitoring.



Recommended health states:



```text

healthy

degraded

unhealthy

unknown

```



Suggested HTTP endpoints of the healthcheck service:



```text

GET /health

GET /ready

GET /metrics

```



Expected behavior:



\* `/health` returns process health.

\* `/ready` returns readiness of the complete LiteLLM -> LM Studio path.

\* `/metrics` exposes Prometheus-compatible metrics if Prometheus is used.



\### 5.2. Model lifecycle controller



Implement a controller responsible for predictable model loading and unloading.



Responsibilities:



\* load the primary model on startup;

\* optionally run a warmup prompt;

\* verify that the expected model is available;

\* unload idle models if policy requires it;

\* reload model after failure;

\* expose current loaded-model state.



The controller must not assume that LM Studio always keeps the model loaded. LM Studio may use JIT loading, Idle TTL, and Auto-Evict.



Suggested configuration:



```yaml

models:

&#x20; primary:

&#x20;   public\_name: local-main

&#x20;   lmstudio\_id: <lmstudio-model-id>

&#x20;   preload: true

&#x20;   warmup: true

&#x20;   ttl\_seconds: 3600

&#x20;   max\_context\_tokens: 8192

```



\### 5.3. Request queue and concurrency limiter



Implement a backpressure layer if LiteLLM and LM Studio are used for more than one interactive user.



The goal is to protect LM Studio from overload.



Required behavior:



\* limit max concurrent requests per model;

\* limit max queued requests;

\* reject excess requests with HTTP `429`;

\* apply queue timeout;

\* expose queue length and active request count;

\* optionally support priority classes.



Suggested default policy:



```yaml

concurrency:

&#x20; default\_max\_active\_requests: 1

&#x20; default\_max\_queued\_requests: 16

&#x20; queue\_timeout\_seconds: 30

&#x20; reject\_status\_code: 429

```



This layer may be implemented as:



1\. middleware in front of LiteLLM;

2\. sidecar proxy;

3\. custom FastAPI service that forwards requests to LiteLLM;

4\. reverse proxy plugin if the selected reverse proxy supports it.



For the first implementation, prefer a simple FastAPI sidecar.



\### 5.4. Metrics and logging



Collect at least the following metrics:



```text

llm\_requests\_total

llm\_request\_errors\_total

llm\_request\_latency\_seconds

llm\_time\_to\_first\_token\_seconds

llm\_tokens\_per\_second

llm\_input\_tokens\_total

llm\_output\_tokens\_total

llm\_queue\_length

llm\_active\_requests

llm\_backend\_health

llm\_loaded\_models

```



Also collect system metrics:



```text

gpu\_memory\_used\_bytes

gpu\_memory\_total\_bytes

gpu\_utilization\_ratio

ram\_used\_bytes

ram\_total\_bytes

process\_cpu\_seconds\_total

```



Preferred stack:



```text

Prometheus

Grafana

node\_exporter

nvidia\_dcgm\_exporter or nvidia-smi based exporter

```



Logs must include:



\* request id;

\* public model name;

\* backend model id;

\* latency;

\* status code;

\* error type;

\* token counts if available;

\* backend health state.



Do not log full prompts or completions by default. Add an explicit configuration flag if prompt logging is required for debugging.



\### 5.5. Compatibility tests



Implement automated tests for API behavior.



Required test cases:



1\. `/v1/chat/completions` non-streaming request.

2\. `/v1/chat/completions` streaming request.

3\. `/v1/responses` request if the selected client uses Responses API.

4\. `system` message behavior.

5\. `temperature`, `top\_p`, `max\_tokens`, `stop` handling.

6\. too-large prompt behavior.

7\. backend unavailable behavior.

8\. queue overflow behavior.

9\. timeout behavior.

10\. model alias resolution through LiteLLM.



If the selected model does not support tools, structured output, or system messages, tests must document this explicitly instead of assuming OpenAI parity.



\## 6. Non-goals



The first version must not implement:



\* custom LLM inference runtime;

\* GPU batch scheduler;

\* tensor parallelism;

\* prefill/decode disaggregation;

\* replacement for vLLM/SGLang;

\* full multi-tenant SaaS billing;

\* public internet exposure of LM Studio;

\* prompt storage by default;

\* model fine-tuning.



\## 7. Expected repository structure



```text

.

├── README.md

├── PROJECT\_CONTEXT.md

├── docker-compose.yml

├── .env.example

├── config

│   ├── litellm.config.yaml

│   ├── gateway.yaml

│   └── models.yaml

├── services

│   ├── healthcheck

│   │   ├── app

│   │   │   ├── main.py

│   │   │   ├── settings.py

│   │   │   ├── lmstudio\_client.py

│   │   │   └── metrics.py

│   │   ├── tests

│   │   └── Dockerfile

│   ├── lifecycle

│   │   ├── app

│   │   │   ├── main.py

│   │   │   ├── settings.py

│   │   │   └── controller.py

│   │   ├── tests

│   │   └── Dockerfile

│   └── queue\_proxy

│       ├── app

│       │   ├── main.py

│       │   ├── settings.py

│       │   ├── limiter.py

│       │   └── forwarder.py

│       ├── tests

│       └── Dockerfile

├── scripts

│   ├── smoke\_test.sh

│   ├── warmup\_model.sh

│   └── list\_models.sh

├── observability

│   ├── prometheus.yml

│   └── grafana

│       └── dashboards

└── tests

&#x20;   ├── integration

&#x20;   └── compatibility

```



\## 8. Environment variables



```bash

\# LiteLLM

LITELLM\_MASTER\_KEY=change-me

DATABASE\_URL=postgresql://litellm:litellm@postgres:5432/litellm

REDIS\_HOST=redis

REDIS\_PORT=6379



\# LM Studio

LMSTUDIO\_BASE\_URL=http://lmstudio:1234

LMSTUDIO\_OPENAI\_BASE\_URL=http://lmstudio:1234/v1

LMSTUDIO\_MODEL\_ID=<lmstudio-model-id>



\# Gateway

PUBLIC\_MODEL\_NAME=local-main

REQUEST\_TIMEOUT\_SECONDS=120

MAX\_ACTIVE\_REQUESTS=1

MAX\_QUEUED\_REQUESTS=16

QUEUE\_TIMEOUT\_SECONDS=30



\# Observability

ENABLE\_PROMPT\_LOGGING=false

ENABLE\_PROMETHEUS=true

LOG\_LEVEL=INFO

```



\## 9. Docker Compose target



The initial `docker-compose.yml` should run:



\* LiteLLM Proxy;

\* Postgres;

\* Redis;

\* healthcheck service;

\* lifecycle controller;

\* optional queue proxy;

\* Prometheus;

\* Grafana.



LM Studio may run outside Docker on the host machine because GPU and desktop/headless configuration can be OS-specific.



When LM Studio runs on the host, Docker services must reach it through a configurable host address.



Examples:



```bash

LMSTUDIO\_OPENAI\_BASE\_URL=http://host.docker.internal:1234/v1

```



or, on Linux if `host.docker.internal` is not available:



```bash

LMSTUDIO\_OPENAI\_BASE\_URL=http://172.17.0.1:1234/v1

```



\## 10. Minimal LiteLLM configuration



```yaml

model\_list:

&#x20; - model\_name: local-main

&#x20;   litellm\_params:

&#x20;     model: openai/${LMSTUDIO\_MODEL\_ID}

&#x20;     api\_base: ${LMSTUDIO\_OPENAI\_BASE\_URL}

&#x20;     api\_key: lm-studio

&#x20;     timeout: 120

&#x20;     supports\_system\_message: true



general\_settings:

&#x20; master\_key: ${LITELLM\_MASTER\_KEY}

&#x20; database\_url: ${DATABASE\_URL}



router\_settings:

&#x20; routing\_strategy: simple-shuffle

&#x20; redis\_host: ${REDIS\_HOST}

&#x20; redis\_port: ${REDIS\_PORT}

```



If the selected model behaves incorrectly with `system` messages, set:



```yaml

supports\_system\_message: false

```



\## 11. Smoke test



A basic smoke test must verify the public gateway path:



```bash

curl http://localhost:4000/v1/chat/completions \\

&#x20; -H "Authorization: Bearer ${LITELLM\_MASTER\_KEY}" \\

&#x20; -H "Content-Type: application/json" \\

&#x20; -d '{

&#x20;   "model": "local-main",

&#x20;   "messages": \[

&#x20;     {

&#x20;       "role": "user",

&#x20;       "content": "Return exactly: ok"

&#x20;     }

&#x20;   ],

&#x20;   "temperature": 0,

&#x20;   "max\_tokens": 8

&#x20; }'

```



Expected result:



\* HTTP status `200`;

\* valid OpenAI-compatible JSON response;

\* response text contains `ok` or an equivalent successful completion;

\* metrics are updated.



\## 12. Implementation priorities



\### Phase 1: Minimal working gateway



1\. Create `docker-compose.yml`.

2\. Add LiteLLM config.

3\. Connect LiteLLM to external LM Studio.

4\. Add `.env.example`.

5\. Add smoke test script.

6\. Document how to start LM Studio or `llmster`.



\### Phase 2: Reliability



1\. Add healthcheck service.

2\. Add lifecycle controller.

3\. Add model warmup.

4\. Add backend readiness checks.

5\. Add integration tests.



\### Phase 3: Backpressure



1\. Add queue proxy.

2\. Implement per-model concurrency limit.

3\. Implement queue timeout.

4\. Implement `429` rejection on overload.

5\. Export queue metrics.



\### Phase 4: Observability



1\. Add Prometheus.

2\. Add Grafana dashboard.

3\. Add structured logs.

4\. Add GPU/RAM metrics.

5\. Add latency and token metrics.



\### Phase 5: Hardening



1\. Add reverse proxy.

2\. Add TLS.

3\. Add IP allowlist or external auth if needed.

4\. Disable prompt logging by default.

5\. Add backup policy for Postgres.

6\. Add failure-mode tests.



\## 13. Coding standards



General requirements:



\* Prefer Python 3.12 for custom services.

\* Prefer FastAPI for HTTP services.

\* Use `pydantic-settings` for configuration.

\* Use `httpx` for async HTTP calls.

\* Use structured JSON logging.

\* Keep services small and independently testable.

\* Add type hints to all public functions.

\* Add unit tests for non-trivial logic.

\* Do not hardcode model identifiers, ports, API keys, or hostnames.

\* Read configuration from environment variables or YAML files.



Error handling requirements:



\* Use explicit timeout values.

\* Return deterministic error responses.

\* Distinguish backend unavailable, queue overflow, invalid request, and internal errors.

\* Do not swallow exceptions silently.

\* Do not retry indefinitely.



Security requirements:



\* Do not log API keys.

\* Do not log prompts unless `ENABLE\_PROMPT\_LOGGING=true`.

\* Do not expose LM Studio directly outside the trusted network.

\* Do not commit `.env`.

\* Provide `.env.example`.



\## 14. Key assumptions



\* LM Studio is already installed and can serve a selected local model.

\* The selected LM Studio model identifier is known.

\* LiteLLM is the only public LLM API endpoint exposed to clients.

\* LM Studio is treated as a private backend.

\* The first version targets one primary local model.

\* Multi-model support may be added later.

\* High-throughput GPU serving is out of scope for the first version.



\## 15. Future migration path



If LM Studio becomes the bottleneck, keep LiteLLM as the public gateway and replace the backend:



```text

LiteLLM Proxy

&#x20; -> vLLM

```



or:



```text

LiteLLM Proxy

&#x20; -> SGLang

```



The public model name should remain stable so clients do not need to change their configuration.



\## 16. Definition of done for MVP



The MVP is complete when:



1\. A client can send an OpenAI-compatible request to LiteLLM.

2\. LiteLLM forwards the request to LM Studio.

3\. LM Studio returns a valid response.

4\. A smoke test passes from the command line.

5\. Configuration is environment-driven.

6\. Basic healthcheck endpoint exists.

7\. README contains startup instructions.

8\. No secrets are committed.

9\. LM Studio is not publicly exposed.

10\. Failure of LM Studio is reported clearly.



