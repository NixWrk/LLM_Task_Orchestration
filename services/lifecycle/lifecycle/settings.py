from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "lifecycle"
    config_path: str = Field(
        default="/app/config/orchestrator.yaml",
        validation_alias="LIFECYCLE_CONFIG_PATH",
    )
    registry_path: str = Field(
        default="/app/state/backend_registry.json",
        validation_alias="BACKEND_REGISTRY_PATH",
    )
    gpu_inventory_url: str = "http://gpu-inventory:4200"
    dry_run: bool = Field(default=True, validation_alias="LIFECYCLE_DRY_RUN")
    enable_reconcile_loop: bool = Field(
        default=False,
        validation_alias="LIFECYCLE_ENABLE_RECONCILE_LOOP",
    )
    reconcile_interval_seconds: float = Field(
        default=15.0,
        gt=0,
        validation_alias="LIFECYCLE_RECONCILE_INTERVAL_SECONDS",
    )
    docker_binary: str = Field(default="docker", validation_alias="LIFECYCLE_DOCKER_BINARY")
    request_timeout_seconds: float = Field(default=5.0, gt=0)
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
