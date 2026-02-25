"""Configuration management for noid-rag."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


class ParserConfig(BaseModel):
    ocr_enabled: bool = True
    ocr_engine: str = "easyocr"
    max_pages: int = 0  # 0 = no limit


class ChunkerConfig(BaseModel):
    method: Literal["hybrid", "fixed"] = "hybrid"
    max_tokens: int = 512
    tokenizer: str = "BAAI/bge-small-en-v1.5"
    overlap: int = 50  # for fixed method


class EmbeddingConfig(BaseModel):
    provider: str = "openrouter"
    api_url: str = "https://openrouter.ai/api/v1/embeddings"
    api_key: SecretStr = SecretStr("")
    model: str = "openai/text-embedding-3-small"
    batch_size: int = 64


class VectorStoreConfig(BaseModel):
    dsn: str = ""
    table_name: str = "documents"
    embedding_dim: int = 1536
    pool_size: int = 20
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    fts_language: str = "english"

    @field_validator("table_name")
    @classmethod
    def _validate_table_name(cls, v: str) -> str:
        """Reject table names that would allow SQL injection."""
        if not _SAFE_IDENTIFIER_RE.match(v):
            raise ValueError(
                f"table_name {v!r} is not a safe SQL identifier. "
                "Use only letters, digits, and underscores, starting with a letter or underscore."
            )
        return v

    @field_validator("fts_language")
    @classmethod
    def _validate_fts_language(cls, v: str) -> str:
        """Reject fts_language values that could corrupt the regconfig cast."""
        if not _SAFE_IDENTIFIER_RE.match(v):
            raise ValueError(
                f"fts_language {v!r} is not a valid PostgreSQL text-search configuration name. "
                "Use only letters, digits, and underscores, starting with a letter or underscore "
                "(e.g. 'english', 'spanish', 'simple'). "
                "Schema-qualified names (e.g. 'pg_catalog.english') are not supported."
            )
        return v


class SearchConfig(BaseModel):
    top_k: int = Field(default=5, gt=0)
    rrf_k: int = Field(default=60, gt=0)


class LLMConfig(BaseModel):
    api_url: str = "https://openrouter.ai/api/v1/chat/completions"
    api_key: SecretStr = SecretStr("")
    model: str = "openai/gpt-4o-mini"
    max_tokens: int = 1024
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    system_prompt: str = (
        "You are a helpful assistant. Answer the user's question based ONLY on the "
        "provided context. If the context doesn't contain enough information to answer, "
        "say so clearly. Be concise and direct."
    )


class BatchConfig(BaseModel):
    max_retries: int = 3
    retry_min_wait: float = 1.0
    retry_max_wait: float = 60.0
    continue_on_error: bool = True
    history_dir: str = "~/.noid-rag/history"


class GenerateConfig(BaseModel):
    model: str = "google/gemini-3.1-pro-preview"
    num_questions: int = 20
    questions_per_chunk: int = 3
    strategy: Literal["random", "diverse"] = "diverse"
    max_tokens: int = 2048


class EvalConfig(BaseModel):
    backend: Literal["ragas", "promptfoo"] = "ragas"
    metrics: list[str] = ["faithfulness", "answer_relevancy", "context_precision"]
    save_results: bool = True
    results_dir: str = "~/.noid-rag/eval"
    promptfoo_threshold: float = 0.7
    judge_max_tokens: int = Field(default=16, gt=0)
    judge_temperature: float = Field(default=0.0, ge=0.0, le=2.0)


def _load_yaml_config(path: Path | None = None) -> dict[str, Any]:
    """Load config from YAML file."""
    if path is None:
        path = Path.home() / ".noid-rag" / "config.yml"
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NOID_RAG_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    parser: ParserConfig = Field(default_factory=ParserConfig)
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    generate: GenerateConfig = Field(default_factory=GenerateConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    config_file: Path | None = None
    verbose: bool = False

    @classmethod
    def load(cls, config_file: Path | None = None, **overrides: Any) -> Settings:
        """Load settings with YAML + env + overrides."""
        yaml_data = _load_yaml_config(config_file)
        merged = {**yaml_data, **{k: v for k, v in overrides.items() if v is not None}}
        if config_file:
            merged["config_file"] = config_file
        return cls(**merged)
