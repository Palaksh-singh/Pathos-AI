"""
Pathos AI — Configuration & Environment Validation
====================================================
Centralized, strictly-typed application settings using pydantic-settings.
Fails fast (at import time) if required secrets are missing in production,
rather than surfacing cryptic errors mid-request.
"""
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class VectorBackend(str, Enum):
    QDRANT = "qdrant"
    PINECONE = "pinecone"


class Settings(BaseSettings):
    """
    All runtime configuration for Pathos AI. Values are sourced from
    environment variables / a local .env file (never committed) and
    validated at process startup.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Core app metadata -------------------------------------------------
    app_name: str = "Pathos AI"
    environment: Environment = Environment.LOCAL
    debug: bool = False
    api_v1_prefix: str = "/api/v1"

    # --- Security ------------------------------------------------------------
    jwt_secret_key: SecretStr = Field(
        default=SecretStr("CHANGE_ME_DEV_ONLY_INSECURE_KEY"),
        description="HS256 signing key for JWT access/refresh tokens.",
    )
    jwt_algorithm: Literal["HS256", "HS384", "HS512"] = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    password_hash_scheme: str = "bcrypt"
    allowed_cors_origins_raw: str = Field(
        default="http://localhost:5173",
        alias="allowed_cors_origins",
        description="Comma-separated list of allowed CORS origins.",
    )

    # --- Database ------------------------------------------------------------
    database_url: str = "sqlite+aiosqlite:///./pathos_ai.db"
    database_echo: bool = False

    # --- LLM Provider ----------------------------------------------------------
    llm_provider: LLMProvider = LLMProvider.OPENAI
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None  # e.g. https://api.groq.com/openai/v1 for a free Groq key
    anthropic_api_key: SecretStr | None = None
    generation_model: str = "gpt-4o-mini"
    generation_temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    generation_max_output_tokens: int = 900
    guardrail_judge_model: str = "gpt-4o-mini"

    # --- Vector store / retrieval -----------------------------------------------
    vector_backend: VectorBackend = VectorBackend.QDRANT
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "pathos_clinical_kb"
    pinecone_api_key: SecretStr | None = None
    pinecone_index: str = "pathos-clinical-kb"
    embedding_provider: Literal["openai", "local"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    local_embedding_dimensions: int = 384
    hybrid_dense_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    retrieval_top_k_dense: int = 12
    retrieval_top_k_sparse: int = 12
    rerank_top_k: int = 5
    reranker_model: str = "ms-marco-MiniLM-L-12-v2"
    max_context_tokens: int = 2200  # hard cap to prevent context stuffing

    # --- Observability ---------------------------------------------------------
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "pathos-ai"
    otel_exporter_endpoint: str | None = None
    otel_service_name: str = "pathos-ai-backend"

    # --- Guardrails / clinical safety --------------------------------------------
    max_regeneration_retries: int = 1
    enable_crisis_detection: bool = True
    disclaimer_text: str = (
        "Pathos AI provides general educational information only and does not "
        "diagnose conditions, prescribe treatment, or replace professional "
        "medical judgment. For any urgent or worsening symptoms, contact a "
        "licensed clinician or emergency services immediately."
    )

    # --- Rate limiting -----------------------------------------------------------
    rate_limit_requests_per_minute: int = 30

    @property
    def allowed_cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_cors_origins_raw.split(",") if origin.strip()]

    @property
    def active_embedding_dimensions(self) -> int:
        """The vector size Qdrant's collection must be created with — depends
        on whether embeddings come from OpenAI or the local model."""
        return self.local_embedding_dimensions if self.embedding_provider == "local" else self.embedding_dimensions

    @model_validator(mode="after")
    def _validate_provider_keys(self) -> "Settings":
        """Fail fast in non-local/test environments if required secrets are absent."""
        if self.environment in (Environment.STAGING, Environment.PRODUCTION):
            if self.jwt_secret_key.get_secret_value() == "CHANGE_ME_DEV_ONLY_INSECURE_KEY":
                raise ValueError(
                    "PATHOS_AI_JWT_SECRET_KEY must be overridden outside local/test environments."
                )
            if self.llm_provider == LLMProvider.OPENAI and not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")
            if self.llm_provider == LLMProvider.ANTHROPIC and not self.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic.")
        return self

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — environment is read once per process."""
    return Settings()


settings = get_settings()