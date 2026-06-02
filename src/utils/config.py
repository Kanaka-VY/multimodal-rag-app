from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    cohere_api_key: str = ""
    llm_provider: str = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    clip_model: str = "sentence-transformers/clip-ViT-B-32"
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache
def get_model_config() -> dict[str, Any]:
    return load_yaml(ROOT / "configs" / "model_config.yaml")


@lru_cache
def get_db_config() -> dict[str, Any]:
    return load_yaml(ROOT / "configs" / "db_config.yaml")
