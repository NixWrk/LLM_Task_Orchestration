# План доработок до LLM-оркестратора

## Цель

Целевая система должна быть локальным LLM-оркестратором для сервера с несколькими GPU и несколькими внутренними сервисами-потребителями.

Она должна автоматически регулировать:

- количество запущенных экземпляров моделей;
- размещение моделей по GPU и backend runtime;
- количество параллельных запросов к каждой модели;
- очередь запросов к каждой модели;
- лимиты input/output/total tokens на каждый запрос;
- деградацию, отказ и восстановление backend-моделей.

Оркестратор не должен сам быть inference runtime. Он должен управлять runtime-ами вроде LM Studio, vLLM, SGLang или других OpenAI-compatible backend-ов.

## Текущее состояние

Уже реализовано:

- `Queue Proxy` как публичный OpenAI-compatible вход на `:4100`.
- Per-model active request limiter.
- Per-model queue size и queue timeout.
- Примерная оценка input tokens по длине текста.
- `default_max_output_tokens`, `max_output_tokens`, `max_input_tokens`, `max_total_tokens`.
- Возврат `429` при переполнении или таймауте очереди.
- Возврат `413` при превышении token budget.
- LiteLLM как upstream gateway на `:4000`.
- LM Studio как первый backend.
- Healthcheck service.
- Prometheus scrape config.
- Docker Compose для локального запуска.
- Unit-тесты для limiter и token policy.
- Fake OpenAI-compatible backend для repeatable integration tests.
- Integration tests для non-streaming, streaming, token budget rejection, queue overflow, queue timeout и upstream unavailable.

Главный пробел:

- пока нет control plane, который реально запускает, останавливает, прогревает и размещает модели по GPU.

## Архитектурная модель

Рекомендуемая модель состоит из двух слоев.

### Data Plane

Быстрый путь запроса:

```text
service client
  -> queue-proxy
      -> token policy
      -> queue/concurrency limiter
      -> backend router
      -> LiteLLM or direct backend
      -> model runtime
```

Data Plane должен быть максимально предсказуемым: принять, поставить в очередь, отклонить или отправить запрос.

### Control Plane

Фоновое управление:

```text
lifecycle-controller
  -> reads model policies
  -> reads GPU/backend/queue metrics
  -> decides desired replicas
  -> starts/stops/warms model runtimes
  -> publishes backend registry
```

Control Plane должен работать циклом reconcile: сравнить желаемое состояние с фактическим и выполнить недостающие действия.

## Целевые сущности

### Model Policy

Описывает публичную модель:

- public name;
- aliases;
- token limits;
- queue limits;
- priority policy;
- min/max replicas;
- idle TTL;
- allowed GPU list;
- preferred runtime type;
- model artifact/path;
- estimated VRAM.

### Backend Instance

Описывает один запущенный экземпляр модели:

- instance id;
- model name;
- runtime type: `lmstudio`, `vllm`, `sglang`, `openai-compatible`;
- base URL;
- GPU ids;
- state: `starting`, `warming`, `ready`, `draining`, `failed`, `stopped`;
- active requests;
- queue pressure;
- health state;
- last used timestamp.

### GPU State

Описывает карту GPU:

- gpu id;
- name;
- total/free/used VRAM;
- utilization;
- temperature if available;
- running model instances;
- reserved VRAM.

## Этап 1. Production-ready Queue Proxy

Задачи:

1. Добавить полноценную поддержку streaming responses без потери cleanup при disconnect.
2. Разделить лимиты для `/chat/completions`, `/responses`, `/embeddings`.
3. Добавить per-service quotas по API key или service id.
4. Добавить priority classes: `interactive`, `batch`, `maintenance`.
5. Добавить request id и structured JSON logs.
6. Добавить middleware для correlation id.
7. Добавить admin endpoint `/status`, `/policies`, `/queues`.
8. Добавить graceful shutdown: не принимать новые запросы, дождаться активных.

Критерии готовности:

- очередь корректно освобождается при ошибке upstream и disconnect клиента;
- все отказы имеют стабильный JSON формат;
- Prometheus показывает active, queued, rejected, latency по model/service/endpoint;
- есть тесты queue overflow, timeout, disconnect, streaming.

## Этап 2. Backend Registry и Router

Задачи:

1. Ввести backend registry в Redis или Postgres.
2. Описать backend instances: URL, model, runtime, GPU, state, capacity.
3. Научить queue proxy выбирать конкретный backend instance.
4. Добавить routing policy: least-active, weighted, healthy-only.
5. Добавить draining mode для backend-а перед остановкой.
6. Разрешить несколько backend instances для одной public model.

Критерии готовности:

- один public model может иметь несколько backend replicas;
- queue proxy не отправляет запросы в unhealthy/draining backend;
- backend registry можно посмотреть через admin endpoint;
- тестами покрыты fallback и отсутствие healthy backend.

## Этап 3. GPU Inventory

Задачи:

1. Добавить сервис или модуль GPU collector.
2. Собирать данные через `nvidia-smi` или DCGM exporter.
3. Нормализовать GPU ids и names.
4. Хранить текущую карту GPU в registry.
5. Добавить reserved VRAM per model.
6. Добавить health state GPU.

Критерии готовности:

- оркестратор видит все GPU сервера;
- видно free/used/total VRAM;
- scheduler может проверить, помещается ли модель на GPU;
- метрики GPU доступны в Prometheus.

## Этап 4. Lifecycle Controller

Задачи:

1. Создать `services/lifecycle`.
2. Реализовать reconcile loop.
3. Читать `config/orchestrator.yaml`.
4. Поддержать desired state: `min_replicas`, `max_replicas`, `idle_ttl_seconds`.
5. Поддержать runtime adapters:
   - `lmstudio` adapter;
   - `docker-vllm` adapter;
   - `docker-sglang` adapter.
6. Запускать backend instance на конкретном GPU.
7. Прогревать модель warmup-запросом.
8. Переводить backend в `ready` только после healthcheck.
9. Останавливать idle backend после TTL.
10. Перезапускать failed backend с backoff.

Критерии готовности:

- controller держит `min_replicas` запущенными;
- controller запускает дополнительную реплику при давлении очереди;
- controller не запускает модель, если нет VRAM;
- controller останавливает idle модели;
- все переходы состояния видны через API и logs.

## Этап 5. Scheduler

Задачи:

1. Реализовать расчет capacity per GPU.
2. Выбирать GPU по стратегии:
   - enough free VRAM;
   - least loaded;
   - model affinity;
   - anti-affinity для тяжелых моделей.
3. Учитывать estimated VRAM модели.
4. Учитывать max replicas per model.
5. Учитывать global max running models.
6. Добавить cooldown между scale up/down.

Критерии готовности:

- scheduler принимает объяснимые решения;
- decision log показывает причину выбора GPU или отказа;
- есть тесты на placement и нехватку VRAM.

## Этап 6. Token Accounting

Задачи:

1. Заменить грубую оценку символов на tokenizer-aware подсчет.
2. Поддержать разные tokenizer profiles per model.
3. Считать actual usage из OpenAI-compatible response.
4. Экспортировать input/output tokens по service/model.
5. Добавить daily/hourly budgets для сервисов.
6. Добавить режим hard/soft limits.

Критерии готовности:

- token limits близки к реальным ограничениям backend-а;
- расход токенов виден по каждому сервису;
- превышение budget возвращает стабильный `429` или `402/403` по политике.

## Этап 7. Reliability и Failure Modes

Задачи:

1. Добавить интеграционные тесты с fake OpenAI backend.
2. Проверить backend unavailable.
3. Проверить backend slow response.
4. Проверить streaming failure mid-response.
5. Проверить model not loaded.
6. Проверить no healthy backends.
7. Проверить queue pressure scale up.
8. Проверить idle scale down.

Критерии готовности:

- отказ backend-а не ломает очередь;
- запросы получают понятные ошибки;
- lifecycle controller восстанавливает failed replicas;
- smoke и integration tests запускаются одной командой.

## Этап 8. Observability

Задачи:

1. Добавить Grafana dashboards.
2. Добавить метрики lifecycle controller.
3. Добавить метрики backend registry.
4. Добавить request latency, TTFT, tokens/sec.
5. Добавить GPU dashboards.
6. Добавить alert rules:
   - no healthy backend;
   - high queue length;
   - high rejection rate;
   - GPU memory exhaustion;
   - backend restart loop.

Критерии готовности:

- состояние всей системы видно с одного dashboard;
- можно понять, какой сервис создает нагрузку;
- можно понять, какая модель и GPU являются bottleneck.

## Этап 9. Security и Operations

Задачи:

1. Спрятать LiteLLM debug endpoint из public bind.
2. Добавить reverse proxy.
3. Добавить TLS.
4. Добавить service API keys.
5. Добавить IP allowlist.
6. Добавить secret handling через `.env` или secret store.
7. Добавить backup policy для Postgres.
8. Добавить runbooks.

Критерии готовности:

- LM Studio и backend runtime-ы не доступны снаружи;
- только queue proxy является публичным внутренним endpoint;
- secrets не попадают в logs и git;
- есть инструкция восстановления после сбоя.

## Рекомендуемый порядок ближайших работ

1. Довести queue proxy до production-safe состояния: disconnect handling, request ids, stable errors.
2. Сделать backend registry.
3. Подключить router к registry.
4. Сделать GPU inventory через `nvidia-smi`.
5. Сделать lifecycle controller с dry-run режимом.
6. Добавить первый runtime adapter для Docker vLLM.
7. Добавить scaling policy на основе queue pressure.
8. Добавить Grafana dashboards.
9. Закрыть security/ops контур.

## Минимальная целевая версия

Минимальная целевая версия считается готовой, когда:

1. Несколько внутренних сервисов ходят только в queue proxy.
2. Для каждой модели задаются token, queue и concurrency limits.
3. Для одной модели можно иметь несколько backend replicas.
4. Оркестратор видит GPU и свободную VRAM.
5. Lifecycle controller сам держит `min_replicas`.
6. Lifecycle controller поднимает дополнительную replica при давлении очереди.
7. Lifecycle controller останавливает idle модели.
8. Queue proxy не отправляет запросы в unhealthy backend.
9. Все ключевые состояния видны в Prometheus/Grafana.
10. Failure modes покрыты интеграционными тестами.
