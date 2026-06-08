"""
Local dev settings — validated via pydantic-settings.
Auth: AzureCliCredential (az login) for Foundry.
      API key for AI Search (no role assignment needed with Contributor access).

Changes vs original:
  - Added AZURE_COSMOS_* settings block for Cosmos DB storage.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Azure AI Foundry ──────────────────────────────────────────────────────
    AZURE_FOUNDRY_PROJECT_ENDPOINT: AnyHttpUrl
    AZURE_OPENAI_CHAT_DEPLOYMENT: str      = "gpt-4o"
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = "text-embedding-ada-002"
    AZURE_OPENAI_API_VERSION: str          = "2024-08-01-preview"

    # ── Azure AI Search — single index, API key ───────────────────────────────
    AZURE_SEARCH_ENDPOINT: AnyHttpUrl
    AZURE_SEARCH_API_KEY: SecretStr
    AZURE_SEARCH_INDEX: str               = "idx-rag"
    AZURE_SEARCH_SEMANTIC_CONFIG: str     = "rag-semantic-config"

    # ── Azure Cosmos DB ───────────────────────────────────────────────────────
    # Portal: Cosmos DB account → Overview → URI
    AZURE_COSMOS_ENDPOINT: str            = ""
    # Portal: Cosmos DB → Keys → Primary Key  (local dev only; prod uses Managed Identity)
    AZURE_COSMOS_KEY: SecretStr           = SecretStr("")
    AZURE_COSMOS_DATABASE: str            = "rag-enterprise"
    # Container names — created automatically on first run if they don't exist
    AZURE_COSMOS_CONTAINER_CONVERSATIONS: str = "conversations"
    AZURE_COSMOS_CONTAINER_FEEDBACK: str      = "feedback"
    AZURE_COSMOS_CONTAINER_MEMORY: str        = "memory"

    # ── RAG tuning ────────────────────────────────────────────────────────────
    CONFIDENCE_THRESHOLD: float  = Field(default=0.75, ge=0.0, le=1.0)
    MAX_RETRIEVAL_ATTEMPTS: int  = Field(default=3,    ge=1,   le=5)
    RETRIEVAL_TOP_K: int         = Field(default=5,    ge=1,   le=20)
    SYNTHESIS_TEMPERATURE: float = Field(default=0.0,  ge=0.0, le=1.0)

    # Short-term memory: how many past turns to inject into synthesis context
    SHORT_TERM_MEMORY_TURNS: int = Field(default=5, ge=1, le=20)

    # ── Observability ─────────────────────────────────────────────────────────
    APPLICATIONINSIGHTS_CONNECTION_STRING: str | None = None
    LOG_LEVEL: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
