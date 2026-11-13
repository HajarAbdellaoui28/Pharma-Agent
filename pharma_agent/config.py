from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", 
        extra="ignore", 
        populate_by_name=True, 
        hide_input_in_errors=True
    )

    api_key: SecretStr = Field(default=SecretStr(""), validation_alias="OPENAI_API_KEY")
    base_url: str | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_BASE_URL", "OPENAI_BASE_UR")
    )
    
    model: str = Field(default="gpt-5.6-terra", validation_alias="OPENAI_MODEL")
    embedding_model: str = Field(
        default="text-embedding-3-small", validation_alias="OPENAI_EMBEDDING_MODEL"
    )
    corpus_dir: Path = Path("corpus")
    cache_dir: Path = Path(".cache/embeddings")
    max_research_rounds: int = Field(default=2, ge=1, le=3)
    max_tool_rounds: int = Field(default=4, ge=1, le=6)
    request_timeout_seconds: float = Field(default=240, ge=10, le=600)
    api_timeout_seconds: float = Field(default=60, ge=1, le=120)
    max_concurrent_requests: int = Field(default=4, ge=1, le=16)

    @field_validator("api_key", "base_url", "model", "embedding_model", mode="before")
    @classmethod
    def remove_env_file_quotes(cls, value):
        # python-dotenv removes matching quotes; Docker --env-file preserves them literally.
        # Accept both transports without changing the supplied credential file.
        if isinstance(value, str):
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
        return value

    @field_validator("base_url")
    @classmethod
    def valid_url(cls, value: str | None) -> str | None:
        if value and not value.startswith(("http://", "https://")):
            raise ValueError("The API base URL must use HTTP or HTTPS")
        return value or None
