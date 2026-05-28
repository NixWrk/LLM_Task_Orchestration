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
    request_timeout_seconds: float = Field(default=120.0, gt=0)
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
