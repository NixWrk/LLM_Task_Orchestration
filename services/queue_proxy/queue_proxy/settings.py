from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "queue-proxy"
    config_path: str = Field(
        default="/app/config/orchestrator.yaml",
        validation_alias="QUEUE_PROXY_CONFIG_PATH",
    )
    upstream_base_url: str = Field(
        default="http://litellm:4000",
        validation_alias="UPSTREAM_LITELLM_BASE_URL",
    )
    upstream_api_key: str = Field(default="", validation_alias="LITELLM_MASTER_KEY")
    backend_registry_url: str = Field(default="", validation_alias="BACKEND_REGISTRY_URL")
    enable_backend_registry_routing: bool = False
    require_backend_registry_backend: bool = False
    queue_proxy_api_key: str = ""
    task_store_backend: str = Field(default="", validation_alias="TASK_STORE_BACKEND")
    task_store_dsn: str = Field(default="", validation_alias="TASK_STORE_DSN")
    task_store_path: str = Field(default="", validation_alias="TASK_STORE_PATH")
    task_executor_enabled: bool = Field(default=False, validation_alias="TASK_EXECUTOR_ENABLED")
    task_executor_interval_seconds: float = Field(
        default=1.0,
        gt=0,
        validation_alias="TASK_EXECUTOR_INTERVAL_SECONDS",
    )
    task_executor_max_attempts: int = Field(
        default=3,
        ge=1,
        validation_alias="TASK_EXECUTOR_MAX_ATTEMPTS",
    )
    task_executor_retry_base_seconds: float = Field(
        default=1.0,
        gt=0,
        validation_alias="TASK_EXECUTOR_RETRY_BASE_SECONDS",
    )
    task_executor_retry_max_seconds: float = Field(
        default=30.0,
        gt=0,
        validation_alias="TASK_EXECUTOR_RETRY_MAX_SECONDS",
    )
    request_timeout_seconds: float = Field(default=120.0, gt=0)
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
