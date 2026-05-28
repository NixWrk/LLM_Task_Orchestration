from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "gpu-inventory"
    nvidia_smi_path: str = "nvidia-smi"
    fake_gpu_inventory_json: str = ""
    command_timeout_seconds: float = Field(default=5.0, gt=0)
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_prefix="GPU_INVENTORY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
