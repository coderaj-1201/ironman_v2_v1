"""
conversation_store.py
=====================
Read / write conversation turns to the Cosmos DB `conversations` container.

Each document (turn) shape:
{
    "id":              "<question_id>",
    "conversation_id": "conv-abc123",       # partition key
    "user_id":         "user-xyz",
    "question_id":     "q-abc123",
    "answer_id":       "ans-def456",
    "query":           "What is the leave policy?",
    "answer":          "Full-time employees are entitled to...",
    "domain":          "hr",
    "confidence":      0.92,
    "sources":         [...],
    "timestamp":       "2024-01-15T10:30:00.000000+00:00"
}

IMPORTANT — Cosmos composite index required for ORDER BY timestamp:
  In Azure Portal → Cosmos DB → your-db → conversations → Settings → Indexing Policy,
  add a composite index:
    [{ "path": "/conversation_id", "order": "ascending" },
     { "path": "/timestamp",       "order": "descending" }]
  provision_cosmos.py sets this automatically via the management SDK.

Gracefully degrades (returns empty / no-ops) when Cosmos is not configured.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _get_container():
    """Lazily resolve container. Returns None if Cosmos not configured."""
    try:
        from shared.azure_clients import get_cosmos_database
        from shared.config import settings
        db = get_cosmos_database()
        if db is None:
            return None
        return db.get_container_client(settings.AZURE_COSMOS_CONTAINER_CONVERSATIONS)
    except Exception as exc:
        logger.warning("conversation_store: cannot get container: %s", exc)
        return None


def save_turn(
    question_id: str,
    answer_id: str,
    conversation_id: str,
    user_id: str,
    query: str,
    answer: str,
    domain: str,
    confidence: float,
    sources: list[dict],
) -> None:
    """
    Upsert a conversation turn.
    Uses question_id as the Cosmos item `id` (unique per question).
    Sync — call via asyncio.to_thread from async code.
    """
    container = _get_container()
    if container is None:
        return

    item: dict[str, Any] = {
        "id":              question_id,
        "conversation_id": conversation_id,
        "user_id":         user_id,
        "question_id":     question_id,
        "answer_id":       answer_id,
        "query":           query,
        "answer":          answer,
        "domain":          domain,
        "confidence":      confidence,
        "sources":         sources,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    }

    try:
        container.upsert_item(item)
        logger.debug("conversation_store: saved turn question_id=%s", question_id)
    except Exception as exc:
        logger.error("conversation_store: failed to save turn %s: %s", question_id, exc)


def get_history(conversation_id: str, limit: int = 10) -> list[dict]:
    """
    Return the last `limit` turns for a conversation, oldest first.
    Sync — call via asyncio.to_thread from async code.

    NOTE: Requires composite index on (conversation_id ASC, timestamp DESC).
    Without it Cosmos raises a 400. provision_cosmos.py sets this automatically.
    Falls back gracefully to an empty list on any error.
    """
    container = _get_container()
    if container is None:
        return []

    try:
        # Parameterised query — OFFSET/LIMIT values are ints, not user strings
        sql = (
            "SELECT c.question_id, c.answer_id, c.query, c.answer, "
            "c.domain, c.confidence, c.timestamp, c.sources "
            "FROM c "
            "WHERE c.conversation_id = @conv_id "
            "ORDER BY c.conversation_id ASC, c.timestamp DESC "
            "OFFSET 0 LIMIT @limit"
        )
        items = list(container.query_items(
            query=sql,
            parameters=[
                {"name": "@conv_id", "value": conversation_id},
                {"name": "@limit",   "value": limit},
            ],
            partition_key=conversation_id,
        ))
        # Reverse so oldest is first (chronological for memory injection)
        items.reverse()
        logger.debug(
            "conversation_store: fetched %d turns for conversation %s",
            len(items), conversation_id,
        )
        return items
    except Exception as exc:
        logger.error("conversation_store: get_history failed: %s", exc)
        return []


async def async_save_turn(
    question_id: str,
    answer_id: str,
    conversation_id: str,
    user_id: str,
    query: str,
    answer: str,
    domain: str,
    confidence: float,
    sources: list[dict],
) -> None:
    """Async wrapper around save_turn."""
    await asyncio.to_thread(
        save_turn,
        question_id, answer_id, conversation_id, user_id,
        query, answer, domain, confidence, sources,
    )


async def async_get_history(conversation_id: str, limit: int = 10) -> list[dict]:
    """Async wrapper around get_history."""
    return await asyncio.to_thread(get_history, conversation_id, limit)
