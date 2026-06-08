"""
Azure client factories — local dev.

Auth:
  - Foundry / OpenAI : AzureCliCredential (az login)
  - AI Search        : API key (no role assignment needed with Contributor access)
  - Cosmos DB        : API key (local dev); Managed Identity in production

lru_cache is fine here — single uvicorn worker in local dev.

Changes vs original:
  - Added get_cosmos_client() factory.
  - Added get_cosmos_database() helper that ensures DB + containers exist.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from azure.ai.projects import AIProjectClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import AzureCliCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from openai import AzureOpenAI

from shared.config import settings

logger = logging.getLogger(__name__)


def _credential() -> AzureCliCredential:
    return AzureCliCredential()


@lru_cache(maxsize=1)
def get_foundry_client() -> AIProjectClient:
    return AIProjectClient(
        endpoint=str(settings.AZURE_FOUNDRY_PROJECT_ENDPOINT),
        credential=_credential(),
    )


@lru_cache(maxsize=1)
def get_openai_client() -> AzureOpenAI:
    return get_foundry_client().get_openai_client(
        api_version=settings.AZURE_OPENAI_API_VERSION
    )


@lru_cache(maxsize=1)
def get_search_client() -> SearchClient:
    """Single index — domain filtered at query time via $filter."""
    return SearchClient(
        endpoint=str(settings.AZURE_SEARCH_ENDPOINT),
        index_name=settings.AZURE_SEARCH_INDEX,
        credential=AzureKeyCredential(settings.AZURE_SEARCH_API_KEY.get_secret_value()),
    )


@lru_cache(maxsize=1)
def get_search_index_client() -> SearchIndexClient:
    return SearchIndexClient(
        endpoint=str(settings.AZURE_SEARCH_ENDPOINT),
        credential=AzureKeyCredential(settings.AZURE_SEARCH_API_KEY.get_secret_value()),
    )


@lru_cache(maxsize=1)
def get_cosmos_client():
    """
    Returns a CosmosClient.
    Local dev: API key auth.
    Production: swap AzureKeyCredential for DefaultAzureCredential / ManagedIdentityCredential.
    Returns None and logs a warning if Cosmos is not configured (optional for local dev
    without Cosmos — agents degrade gracefully).
    """
    try:
        from azure.cosmos import CosmosClient  # type: ignore[import]
        endpoint = settings.AZURE_COSMOS_ENDPOINT
        key = settings.AZURE_COSMOS_KEY.get_secret_value()
        if not endpoint or not key:
            logger.warning(
                "Cosmos DB not configured (AZURE_COSMOS_ENDPOINT / AZURE_COSMOS_KEY missing). "
                "Conversation history, feedback and memory features disabled."
            )
            return None
        return CosmosClient(url=endpoint, credential=key)
    except ImportError:
        logger.warning("azure-cosmos not installed — Cosmos features disabled.")
        return None


def get_cosmos_database():
    """
    Returns the Cosmos database client, creating it if it doesn't exist.
    Also ensures all three containers exist with correct partition keys and TTL.
    Returns None if Cosmos is not configured.
    """
    try:
        from azure.cosmos import PartitionKey, exceptions  # type: ignore[import]
    except ImportError:
        return None

    client = get_cosmos_client()
    if client is None:
        return None

    db_name = settings.AZURE_COSMOS_DATABASE
    try:
        db = client.create_database_if_not_exists(id=db_name)
    except Exception as exc:
        logger.error("Failed to get/create Cosmos database '%s': %s", db_name, exc)
        return None

    # conversations — partition by conversation_id for efficient history reads
    _ensure_container(
        db,
        name=settings.AZURE_COSMOS_CONTAINER_CONVERSATIONS,
        partition_key="/conversation_id",
        default_ttl=None,         # keep indefinitely
    )
    # feedback — partition by user_id
    _ensure_container(
        db,
        name=settings.AZURE_COSMOS_CONTAINER_FEEDBACK,
        partition_key="/user_id",
        default_ttl=None,
    )
    # memory — partition by user_id; short-term items carry their own ttl field
    _ensure_container(
        db,
        name=settings.AZURE_COSMOS_CONTAINER_MEMORY,
        partition_key="/user_id",
        default_ttl=-1,           # -1 = no default TTL; items set their own via 'ttl' field
    )
    return db


def _ensure_container(db, name: str, partition_key: str, default_ttl):
    """Creates the container if it doesn't exist. Silently skips if already there."""
    try:
        from azure.cosmos import PartitionKey  # type: ignore[import]
        kwargs = dict(id=name, partition_key=PartitionKey(path=partition_key))
        if default_ttl is not None:
            kwargs["default_ttl"] = default_ttl
        db.create_container_if_not_exists(**kwargs)
        logger.debug("Cosmos container '%s' ensured.", name)
    except Exception as exc:
        logger.error("Failed to ensure Cosmos container '%s': %s", name, exc)
