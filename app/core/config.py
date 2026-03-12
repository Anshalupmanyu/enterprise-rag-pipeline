"""
Enterprise RAG Pipeline — Application Configuration

Loads all settings from environment variables / .env file using Pydantic Settings.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """Central configuration loaded from environment variables."""

    # ── LLM ──────────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = Field(
        default="http://ollama:11434",
        description="Base URL for the Ollama API server",
    )
    LLM_MODEL: str = Field(
        default="mistral",
        description="Model identifier to use via Ollama",
    )
    LLM_TEMPERATURE: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for generation",
    )

    # ── Embeddings ───────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model for document embeddings",
    )

    # ── Vector Store ─────────────────────────────────────────────────────
    VECTOR_STORE_PATH: str = Field(
        default="./data/vector_store",
        description="Path to persist the FAISS index",
    )
    SIMILARITY_TOP_K: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Number of similar chunks to retrieve",
    )

    # ── Document Chunking ────────────────────────────────────────────────
    CHUNK_SIZE: int = Field(
        default=1000,
        ge=100,
        description="Character count per text chunk",
    )
    CHUNK_OVERLAP: int = Field(
        default=200,
        ge=0,
        description="Overlap between consecutive chunks",
    )

    # ── JWT Authentication ───────────────────────────────────────────────
    JWT_SECRET_KEY: str = Field(
        default="change-me-to-a-random-secret-in-production",
        description="Secret key for signing JWT tokens",
    )
    JWT_ALGORITHM: str = Field(
        default="HS256",
        description="Algorithm used for JWT encoding",
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=60,
        ge=1,
        description="JWT access token lifetime in minutes",
    )

    # ── API ──────────────────────────────────────────────────────────────
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    API_TITLE: str = Field(default="Enterprise RAG Pipeline")
    LOG_LEVEL: str = Field(default="info")

    # ── Documents Path ───────────────────────────────────────────────────
    DOCUMENTS_PATH: str = Field(
        default="./data/documents",
        description="Directory containing raw documents for ingestion",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()
