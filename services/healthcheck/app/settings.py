from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "healthcheck"

    lmstudio_openai_base_url: str = "http://host.docker.internal:1234/v1"
    lmstudio_api_key: str = "lm-studio"
    lmstudio_model_id: str = "local-main"

    litellm_base_url: str = "http://litellm:4000"
    litellm_master_key: str = "sk-change-this-local-key"
    public_model_name: str = "local-main"

    request_timeout_seconds: float = Field(default=120.0, gt=0)
    readiness_prompt: str = "Return exactly: ok"
    readiness_max_tokens: int = Field(default=8, ge=1)
    enable_litellm_path_check: bool = True
    enable_prompt_logging: bool = False
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
