"""
config/settings.py
──────────────────
Centralised application configuration loaded from environment variables
and the .env file via pydantic-settings.

All modules import settings from here rather than reading os.environ directly,
ensuring a single source of truth and compile-time type checking.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings derived from environment variables / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Backend ───────────────────────────────────────────────────────
    llm_backend: str = Field("local", description="'local' or 'openai'")
    local_model_path: Path = Field(
        Path("models/mistral-7b-instruct-v0.3.Q4_K_M.gguf"),
        description="Path to GGUF model file for local inference.",
    )
    openai_api_base: str = Field("https://api.openai.com/v1")
    openai_api_key: str = Field("", description="OpenAI-compatible API key.")
    openai_model: str = Field("gpt-4o-mini")

    # ── Embedding & reranking ────────────────────────────────────────────
    embedding_model: str = Field("sentence-transformers/all-MiniLM-L6-v2")
    reranker_model: str = Field("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ── ChromaDB ─────────────────────────────────────────────────────────
    chroma_persist_dir: Path = Field(Path("data/chroma"))
    chroma_collection_name: str = Field("noia_knowledge_base")

    # ── BM25 ─────────────────────────────────────────────────────────────
    bm25_index_path: Path = Field(Path("data/bm25_index.pkl"))

    # ── Retrieval ─────────────────────────────────────────────────────────
    hybrid_alpha: float = Field(
        0.55,
        ge=0.0,
        le=1.0,
        description="Blend coefficient: 1.0=pure dense, 0.0=pure sparse.",
    )
    retrieval_top_k: int = Field(20, ge=1, le=100)
    rerank_top_n: int = Field(5, ge=1, le=20)
    groundedness_threshold: float = Field(0.78, ge=0.0, le=1.0)

    # ── Chunking ─────────────────────────────────────────────────────────
    chunk_size: int = Field(512, ge=64, le=2048)
    chunk_overlap: int = Field(64, ge=0, le=256)

    # ── API ───────────────────────────────────────────────────────────────
    api_host: str = Field("0.0.0.0")
    api_port: int = Field(8000, ge=1024, le=65535)
    api_secret_key: str = Field("change-me")
    api_algorithm: str = Field("HS256")
    access_token_expire_minutes: int = Field(480, ge=5)

    # ── Corpus ────────────────────────────────────────────────────────────
    corpus_dir: Path = Field(Path("data/corpus"))

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: str = Field("INFO")
    log_format: str = Field("json")

    # ── Audit DB ──────────────────────────────────────────────────────────
    audit_db_path: Path = Field(Path("data/audit.db"))

    @field_validator("llm_backend")
    @classmethod
    def validate_llm_backend(cls, v: str) -> str:
        allowed = {"local", "openai"}
        if v not in allowed:
            raise ValueError(f"llm_backend must be one of {allowed}.")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}.")
        return upper


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
