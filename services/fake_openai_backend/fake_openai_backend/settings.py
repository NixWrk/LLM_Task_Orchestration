from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_ids: str = "local-main"
    response_text: str = "ok"

    model_config = SettingsConfigDict(
        env_prefix="FAKE_OPENAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def models(self) -> list[str]:
        return [model.strip() for model in self.model_ids.split(",") if model.strip()]
