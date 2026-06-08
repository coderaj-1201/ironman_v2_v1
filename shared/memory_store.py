"""
memory_store.py
===============
Short-term and long-term memory for the RAG pipeline.

SHORT-TERM MEMORY
-----------------
Derived from the `conversations` container — no separate store needed.
get_short_term_context() queries the last K turns and formats them as a
readable conversation history string to inject into the synthesis prompt.

LONG-TERM MEMORY
----------------
Stored in the `memory` container.
Each document shape:
{
    "id":          "<memory_id>",     # Cosmos item id
    "memory_id":   "mem-abc123",
    "user_id":     "user-xyz",        # partition key
    "content":     "User prefers answers in bullet points.",
    "memory_type": "long",
    "created_at":  "2024-01-15T10:30:00.000Z",
    "source_question_id": "q-abc123"  # traceability
}

extract_and_save_long_term() calls the LLM to extract saveable facts from
an exchange. It is designed to be fire-and-forget (non-blocking) — failures
do not affect the answer path.

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
_EXTRACT_SYSTEM = """You are a memory extraction assistant.
Given a user question and an assistant answer from an enterprise knowledge system,
extract any reusable facts, preferences, or context about THIS SPECIFIC USER
that would be helpful to remember for future queries.

Rules:
- Only extract facts that are clearly about the user (e.g. their role, department, preferences).
- Do NOT extract general knowledge facts from the answer.
- Return a JSON array of short strings (each < 100 chars). Return [] if nothing to extract.
- No markdown, no explanation.

Example output: ["User is in the HR department", "User prefers concise bullet-point answers"]
"""


# ── Short-term memory ─────────────────────────────────────────────────────────

def get_short_term_context(conversation_id: str, k: int | None = None) -> str:
    """
    Returns the last k turns for a conversation as a formatted string.
    Intended to be prepended to the synthesis prompt.
    Sync — call via asyncio.to_thread.
    Returns empty string if no history or Cosmos unavailable.
    """
    from shared.config import settings
    from shared.conversation_store import get_history

    limit = k or settings.SHORT_TERM_MEMORY_TURNS
    turns = get_history(conversation_id, limit=limit)
    if not turns:
        return ""

    lines = ["[Conversation History — most recent turns]"]
    for t in turns:
        lines.append(f"User: {t.get('query', '')}")
        lines.append(f"Assistant: {t.get('answer', '')}")
    return "\n".join(lines)


async def async_get_short_term_context(conversation_id: str, k: int | None = None) -> str:
    """Async wrapper."""
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


def save_long_term(
    user_id: str,
    content: str,
    source_question_id: str = "",
) -> str | None:
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
        logger.debug("memory_store: saved memory_id=%s user_id=%s", memory_id, user_id)
        return memory_id
    except Exception as exc:
        logger.error("memory_store: failed to save long-term memory: %s", exc)
        return None


def get_long_term(user_id: str) -> list[dict]:
    """
    Retrieve all long-term memory facts for a user.
    Returns list of {memory_id, content, created_at}.
    Sync — call via asyncio.to_thread.
    """
    container = _get_memory_container()
    if container is None:
        return []

    try:
        query = (
            "SELECT c.memory_id, c.content, c.created_at "
            "FROM c WHERE c.user_id = @user_id "
            "ORDER BY c.created_at DESC"
        )
        items = list(container.query_items(
            query=query,
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
    lines = ["[Known user context from previous sessions]"]
    for m in memories:
        lines.append(f"- {m.get('content', '')}")
    return "\n".join(lines)


def extract_facts_from_exchange(query: str, answer: str) -> list[str]:
    """
    Ask the LLM to extract saveable user facts from a Q&A exchange.
    Returns a list of fact strings (may be empty).
    Sync — designed to be called in a background thread.
    """
    from shared.azure_clients import get_openai_client
    from shared.config import settings

    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": f"Question: {query}\n\nAnswer: {answer}"},
            ],
            temperature=0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        raw = json.loads(resp.choices[0].message.content)
        if isinstance(raw, list):
            return [str(f) for f in raw if f]
        # Model may wrap in a key
        for v in raw.values():
            if isinstance(v, list):
                return [str(f) for f in v if f]
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
    Any exception is caught — this must never block the answer path.
    """
    try:
        facts = await asyncio.to_thread(extract_facts_from_exchange, query, answer)
        for fact in facts:
            await asyncio.to_thread(save_long_term, user_id, fact, source_question_id)
        if facts:
            logger.info(
                "memory_store: saved %d long-term facts for user %s", len(facts), user_id
            )
    except Exception as exc:
        logger.warning("memory_store: async_extract_and_save_long_term failed: %s", exc)


async def async_get_long_term_context(user_id: str) -> str:
    """Async wrapper."""
    return await asyncio.to_thread(get_long_term_context, user_id)
