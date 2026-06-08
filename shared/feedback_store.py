"""
feedback_store.py
=================
Write / read feedback records in the Cosmos DB `feedback` container.

Each document shape:
{
    "id":          "<feedback_id>",     # Cosmos item id
    "feedback_id": "fb-abc123",
    "answer_id":   "ans-def456",
    "question_id": "q-ghi789",
    "user_id":     "user-xyz",          # partition key
    "rating":      4,                   # 1–5
    "comment":     "Very helpful",
    "timestamp":   "2024-01-15T10:32:00.000Z"
}

Gracefully degrades when Cosmos is not configured.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _get_container():
    try:
        from shared.azure_clients import get_cosmos_database
        from shared.config import settings
        db = get_cosmos_database()
        if db is None:
            return None
        return db.get_container_client(settings.AZURE_COSMOS_CONTAINER_FEEDBACK)
    except Exception as exc:
        logger.warning("feedback_store: cannot get container: %s", exc)
        return None


def save_feedback(
    feedback_id: str,
    answer_id: str,
    question_id: str,
    user_id: str,
    rating: int,
    comment: str = "",
) -> None:
    """
    Write a feedback record.
    Sync — call via asyncio.to_thread from async code.
    """
    container = _get_container()
    if container is None:
        return

    item: dict[str, Any] = {
        "id":          feedback_id,
        "feedback_id": feedback_id,
        "answer_id":   answer_id,
        "question_id": question_id,
        "user_id":     user_id,
        "rating":      max(1, min(5, rating)),   # clamp to 1–5
        "comment":     comment,
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }

    try:
        container.upsert_item(item)
        logger.info("feedback_store: saved feedback_id=%s rating=%d", feedback_id, rating)
    except Exception as exc:
        logger.error("feedback_store: failed to save %s: %s", feedback_id, exc)


def get_feedback_for_answer(answer_id: str) -> list[dict]:
    """
    Return all feedback records for a given answer_id.
    Cross-partition query — fine for low-volume admin use.
    Sync — call via asyncio.to_thread from async code.
    """
    container = _get_container()
    if container is None:
        return []

    try:
        query = (
            "SELECT * FROM c WHERE c.answer_id = @answer_id ORDER BY c.timestamp DESC"
        )
        items = list(container.query_items(
            query=query,
            parameters=[{"name": "@answer_id", "value": answer_id}],
            enable_cross_partition_query=True,
        ))
        return items
    except Exception as exc:
        logger.error("feedback_store: get_feedback_for_answer failed: %s", exc)
        return []


async def async_save_feedback(
    feedback_id: str,
    answer_id: str,
    question_id: str,
    user_id: str,
    rating: int,
    comment: str = "",
) -> None:
    """Async wrapper — fire-and-forget friendly."""
    await asyncio.to_thread(
        save_feedback, feedback_id, answer_id, question_id, user_id, rating, comment
    )


async def async_get_feedback_for_answer(answer_id: str) -> list[dict]:
    """Async wrapper."""
    return await asyncio.to_thread(get_feedback_for_answer, answer_id)
