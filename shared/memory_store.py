"""
memory_store.py
===============
Short-term and long-term memory for the RAG pipeline.

SHORT-TERM MEMORY
-----------------
Derived from the `conversations` container — no separate store.
get_short_term_context() returns the last K turns as a formatted string
for injection into the synthesis prompt.

LONG-TERM MEMORY
----------------
Stored in the `memory` container. Schema:
{
    "id":                 "<memory_id>",
    "memory_id":          "mem-abc123",
    "user_id":            "user-xyz",      # partition key
    "content":            "User is in the HR department.",
    "memory_type":        "long",
    "created_at":         "2024-01-15T10:30:00.000000+00:00",
    "source_question_id": "q-abc123"
}

extract_facts_from_exchange() prompts the LLM to extract user-specific facts.
Designed to be fire-and-forget — failures never block the answer path.

Gracefully degrades when Cosmos is not configured.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

# ── Long-term extraction prompt ───────────────────────────────────────────────
# NOTE: We use a plain completion (no response_format=json_object) here because
# json_object mode requires the model to return a JSON *object* (dict), not an
# array. Asking for an array with json_object mode causes the model to wrap it,
# which we then have to unwrap. Instead we parse manually and fall back safely.
_EXTRACT_SYSTEM = """You are a memory extraction assistant for an enterprise knowledge system.
Given a user question and assistant answer, extract any reusable facts or preferences
about THIS SPECIFIC USER that would help answer future queries.

Rules:
- Only extract facts clearly about the user (role, department, location, preferences).
- Do NOT extract general knowledge facts from the answer content.
- Return ONLY a raw JSON array of short strings (each under 100 chars).
- If nothing to extract, return exactly: []
- No markdown fences, no keys, no explanation — just the array.

Example: ["User works in the HR department", "User prefers bullet-point summaries"]"""


# ── Short-term memory ─────────────────────────────────────────────────────────

def get_short_term_context(conversation_id: str, k: int | None = None) -> str:
    """
    Returns the last k turns as a formatted conversation history string.
    Returns empty string if no history or Cosmos unavailable.
    Sync — call via asyncio.to_thread.
    """
    from shared.config import settings
    from shared.conversation_store import get_history

    limit = k or settings.SHORT_TERM_MEMORY_TURNS
    turns = get_history(conversation_id, limit=limit)
    if not turns:
        return ""

    lines = ["[Conversation history — use this to resolve follow-up questions]"]
    for t in turns:
        lines.append(f"User: {t.get('query', '')}")
        lines.append(f"Assistant: {t.get('answer', '')}")
    return "\n".join(lines)


async def async_get_short_term_context(conversation_id: str, k: int | None = None) -> str:
    return await asyncio.to_thread(get_short_term_context, conversation_id, k)


# ── Long-term memory ──────────────────────────────────────────────────────────

def _get_memory_container():
    try:
        from shared.azure_clients import get_cosmos_database
        from shared.config import settings
        db = get_cosmos_database()
        if db is None:
            return None
        return db.get_container_client(settings.AZURE_COSMOS_CONTAINER_MEMORY)
    except Exception as exc:
        logger.warning("memory_store: cannot get memory container: %s", exc)
        return None


def save_long_term(user_id: str, content: str, source_question_id: str = "") -> str | None:
    """
    Save a long-term memory fact for a user.
    Returns memory_id on success, None on failure.
    Sync — call via asyncio.to_thread.
    """
    container = _get_memory_container()
    if container is None:
        return None

    memory_id = f"mem-{uuid4().hex[:8]}"
    item: dict[str, Any] = {
        "id":                 memory_id,
        "memory_id":          memory_id,
        "user_id":            user_id,
        "content":            content,
        "memory_type":        "long",
        "created_at":         datetime.now(timezone.utc).isoformat(),
        "source_question_id": source_question_id,
    }

    try:
        container.upsert_item(item)
        logger.debug("memory_store: saved memory_id=%s user=%s", memory_id, user_id)
        return memory_id
    except Exception as exc:
        logger.error("memory_store: failed to save long-term memory: %s", exc)
        return None


def get_long_term(user_id: str) -> list[dict]:
    """
    Retrieve all long-term memory facts for a user, newest first.
    Sync — call via asyncio.to_thread.

    NOTE: Requires composite index on (user_id ASC, created_at DESC).
    provision_cosmos.py sets this automatically.
    """
    container = _get_memory_container()
    if container is None:
        return []

    try:
        sql = (
            "SELECT c.memory_id, c.content, c.created_at "
            "FROM c "
            "WHERE c.user_id = @user_id "
            "ORDER BY c.user_id ASC, c.created_at DESC"
        )
        items = list(container.query_items(
            query=sql,
            parameters=[{"name": "@user_id", "value": user_id}],
            partition_key=user_id,
        ))
        return items
    except Exception as exc:
        logger.error("memory_store: get_long_term failed for user %s: %s", user_id, exc)
        return []


def get_long_term_context(user_id: str) -> str:
    """
    Returns long-term memories as a formatted string for prompt injection.
    Returns empty string if no memories or Cosmos unavailable.
    Sync — call via asyncio.to_thread.
    """
    memories = get_long_term(user_id)
    if not memories:
        return ""
    lines = ["[Known context about this user from previous sessions]"]
    for m in memories:
        lines.append(f"- {m.get('content', '')}")
    return "\n".join(lines)


def extract_facts_from_exchange(query: str, answer: str) -> list[str]:
    """
    Ask the LLM to extract saveable user-specific facts from a Q&A exchange.
    Returns a list of fact strings (may be empty).
    Sync — designed to run in a background thread via asyncio.to_thread.

    Uses plain text completion (not response_format=json_object) because the
    target is a JSON array, and json_object mode requires a dict — causing
    the model to wrap the array unnecessarily.
    """
    from shared.azure_clients import get_openai_client
    from shared.config import settings

    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user",   "content": f"Question: {query}\n\nAnswer: {answer}"},
            ],
            temperature=0,
            max_tokens=200,
            # No response_format=json_object — we want a raw array, not a wrapped dict
        )
        raw_text = resp.choices[0].message.content.strip()

        # Strip markdown fences if model adds them despite instructions
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        parsed = json.loads(raw_text)

        # Accept either a bare array or a wrapped {"facts": [...]} dict
        if isinstance(parsed, list):
            return [str(f) for f in parsed if f and len(str(f)) < 200]
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    return [str(f) for f in v if f and len(str(f)) < 200]
        return []
    except (json.JSONDecodeError, ValueError) as exc:
        logger.debug("memory_store: fact extraction parse error (non-fatal): %s", exc)
        return []
    except Exception as exc:
        logger.warning("memory_store: fact extraction failed: %s", exc)
        return []


async def async_extract_and_save_long_term(
    user_id: str,
    query: str,
    answer: str,
    source_question_id: str = "",
) -> None:
    """
    Fire-and-forget: extract facts from an exchange and persist them.
    Any exception is caught — this must never affect the answer path.
    """
    try:
        facts = await asyncio.to_thread(extract_facts_from_exchange, query, answer)
        for fact in facts:
            await asyncio.to_thread(save_long_term, user_id, fact, source_question_id)
        if facts:
            logger.info(
                "memory_store: saved %d long-term facts for user=%s", len(facts), user_id
            )
    except Exception as exc:
        logger.warning("memory_store: async_extract_and_save_long_term failed: %s", exc)


async def async_get_long_term_context(user_id: str) -> str:
    return await asyncio.to_thread(get_long_term_context, user_id)
