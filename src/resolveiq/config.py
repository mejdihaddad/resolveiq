#resolveiq\src\resolveiq\config.py
import sys
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"no pyproject.toml found above {here}")


REPO_ROOT = _repo_root()
ENV_FILE = REPO_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    gemini_api_key: SecretStr = Field(alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-3.6-flash", alias="GEMINI_MODEL")
    gemini_embed_model: str = Field(default="gemini-embedding-001", alias="GEMINI_EMBED_MODEL")
    embedding_dimensions: int = Field(default=3072, alias="EMBEDDING_DIMENSIONS")
    nyc311_base_url: str = Field(
    default="https://data.cityofnewyork.us/resource", alias="NYC311_BASE_URL"
    )
    nyc311_dataset_id: str = Field(default="erm2-nwe9", alias="NYC311_DATASET_ID")
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_collection: str = Field(
        default="resolveiq-cases-collection-00", alias="QDRANT_COLLECTION"
    )
    qdrant_collection_cases: str = Field(
        default="resolveiq-cases-collection-01", alias="QDRANT_COLLECTION_CASES"
    )
    langfuse_public_key: str | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: SecretStr | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="http://localhost:3000", alias="LANGFUSE_HOST")
    langsmith_tracing: bool = Field(default=False, alias="RESOLVEIQ_LANGSMITH_TRACING")
    langsmith_api_key: SecretStr | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com", alias="LANGSMITH_ENDPOINT"
    )
    langsmith_project: str = Field(default="resolveiq-triage", alias="LANGSMITH_PROJECT")

    @property
    def key_tail(self) -> str:
        value = self.gemini_api_key.get_secret_value()
        return value[-4:] if value else ""


@lru_cache
def get_settings() -> Settings:
    return Settings()


def show_env() -> None:
    settings = get_settings()
    print("config    :", Path(__file__).resolve())
    print("env file  :", ENV_FILE, "found" if ENV_FILE.exists() else "MISSING")
    print("python    :", sys.version.split()[0])
    print("cwd       :", Path.cwd())
    print("key       :", "set, ends " + settings.key_tail if settings.key_tail else "NOT SET")
    print("model     :", settings.gemini_model)
    print("embedder  :", settings.gemini_embed_model, "dims", settings.embedding_dimensions)
    print("qdrant    :", settings.qdrant_url, "collection", settings.qdrant_collection)