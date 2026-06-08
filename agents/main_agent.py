"""
Main Agent — Local Dev
=======================
MAF Functional Workflow (@workflow / @step).
Entry point for all queries. Calls Orchestrator via HTTP.

Changes vs original:
  - /query endpoint: generates question_id, returns it + answer_id in response.
  - After a successful answer: saves conversation turn to Cosmos (fire-and-forget).
  - After a successful answer: extracts + saves long-term memory (fire-and-forget).
  - NEW /feedback  POST endpoint.
  - NEW /history   GET  endpoint.
  - Existing workflow logic is 100% unchanged.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager

import httpx
import uvicorn
from agent_framework import step, workflow
from fastapi import FastAPI, Request, Response

from shared.logging_config import configure_logging, get_logger
from shared.models import FeedbackRequest, FinalResponse, UserQuery

configure_logging()
logger = get_logger(__name__)

_ORCHESTRATOR_URL = "http://localhost:8001"

_FAILURE_MSG = (
    "I wasn't able to find a confident answer after exhausting all retrieval strategies.\n\n"
    "📋 **Option 1 — Raise a Support Ticket**\nReply with: `raise_ticket`\n\n"
    "👤 **Option 2 — Connect with a Subject Matter Expert**\nReply with: `connect_sme`"
)


# ── MAF steps — unchanged from original ──────────────────────────────────────

@step
async def call_orchestrator(user_query: UserQuery) -> FinalResponse:
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{_ORCHESTRATOR_URL}/orchestrate",
            json=user_query.__dict__,
        )
        resp.raise_for_status()
        return FinalResponse(**resp.json())


@step
async def handle_raise_ticket(user_id: str) -> str:
    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    logger.info("Ticket raised ticket_id=%s user_id=%s", ticket_id, user_id)
    return (
        f"✅ **Ticket raised!** Reference: `{ticket_id}`\n"
        f"Expected response: **4 business hours**."
    )


@step
async def handle_connect_sme(user_id: str) -> str:
    logger.info("SME connect requested user_id=%s", user_id)
    return "✅ **Connecting you with an SME.** Expected response: **2 business hours**."


# ── MAF workflow — logic unchanged; Cosmos persistence added as side-effects ──

@workflow(name="main_agent_workflow")
async def main_agent_workflow(user_query: UserQuery) -> str:
    text = user_query.text.strip().lower()

    if text == "raise_ticket":
        return await handle_raise_ticket(user_query.user_id)
    if text == "connect_sme":
        return await handle_connect_sme(user_query.user_id)

    try:
        final: FinalResponse = await call_orchestrator(user_query)
    except Exception as exc:
        logger.error("Orchestrator call failed: %s", exc, exc_info=True)
        return _FAILURE_MSG

    if final.status == "success":
        sources_text = ""
        if final.sources:
            bullets = "\n".join(
                f"  • {s.get('title', s.get('source', ''))}" for s in final.sources
            )
            sources_text = f"\n\n📚 **Sources:**\n{bullets}"
        meta = (
            f"*Domain: {final.domain.upper() if final.domain else 'N/A'} | "
            f"Confidence: {final.confidence:.0%} | Attempts: {final.attempts_used}*"
        )

        # ── Side-effects: Cosmos persistence (fire-and-forget, never blocks answer) ──
        asyncio.create_task(
            _persist_turn(user_query, final)
        )

        return f"{final.answer}{sources_text}\n\n{meta}"

    return _FAILURE_MSG


async def _persist_turn(user_query: UserQuery, final: FinalResponse) -> None:
    """
    Saves conversation turn + triggers long-term memory extraction.
    Runs as a background task — any failure is logged but never surfaced to the user.
    """
    try:
        from shared.conversation_store import async_save_turn
        from shared.memory_store import async_extract_and_save_long_term

        await async_save_turn(
            question_id=user_query.question_id,
            answer_id=final.answer_id,
            conversation_id=user_query.conversation_id,
            user_id=user_query.user_id,
            query=user_query.text,
            answer=final.answer,
            domain=str(final.domain) if final.domain else "",
            confidence=final.confidence,
            sources=final.sources,
        )

        # Long-term memory extraction is genuinely fire-and-forget
        asyncio.create_task(
            async_extract_and_save_long_term(
                user_id=user_query.user_id,
                query=user_query.text,
                answer=final.answer,
                source_question_id=user_query.question_id,
            )
        )
    except Exception as exc:
        logger.warning("_persist_turn failed (non-fatal): %s", exc)


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Main Agent started.")
    yield
    logger.info("Main Agent stopped.")


app = FastAPI(title="RAG Main Agent — Local", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "agent": "main"}


@app.post("/query")
async def query(raw: Request) -> Response:
    body = await raw.json()
    user_query = UserQuery(
        text=body["text"],
        conversation_id=body.get("conversation_id", str(uuid.uuid4())),
        user_id=body.get("user_id", "anonymous"),
        # Allow caller to supply question_id; otherwise auto-generated by UserQuery default
        question_id=body.get("question_id", f"q-{uuid.uuid4().hex[:8]}"),
    )
    result_obj = await main_agent_workflow.run(user_query)
    outputs    = result_obj.get_outputs()
    reply: str = outputs[0] if outputs else _FAILURE_MSG
    return Response(
        content=json.dumps({
            "reply":           reply,
            "question_id":     user_query.question_id,
            "conversation_id": user_query.conversation_id,
        }),
        media_type="application/json",
    )


@app.post("/feedback")
async def feedback(raw: Request) -> Response:
    """
    POST /feedback
    Body: { answer_id, question_id, user_id, rating (1-5), comment? }
    """
    body = await raw.json()
    try:
        fb = FeedbackRequest(
            answer_id=body["answer_id"],
            question_id=body.get("question_id", ""),
            user_id=body.get("user_id", "anonymous"),
            rating=int(body["rating"]),
            comment=body.get("comment", ""),
        )
    except (KeyError, ValueError) as exc:
        return Response(
            content=json.dumps({"error": f"Invalid payload: {exc}"}),
            status_code=400,
            media_type="application/json",
        )

    try:
        from shared.feedback_store import async_save_feedback
        await async_save_feedback(
            feedback_id=fb.feedback_id,
            answer_id=fb.answer_id,
            question_id=fb.question_id,
            user_id=fb.user_id,
            rating=fb.rating,
            comment=fb.comment,
        )
        logger.info(
            "Feedback saved feedback_id=%s answer_id=%s rating=%d",
            fb.feedback_id, fb.answer_id, fb.rating,
        )
    except Exception as exc:
        logger.error("Feedback save failed: %s", exc, exc_info=True)
        return Response(
            content=json.dumps({"error": "Failed to save feedback. Please try again."}),
            status_code=500,
            media_type="application/json",
        )

    return Response(
        content=json.dumps({
            "status":      "saved",
            "feedback_id": fb.feedback_id,
        }),
        media_type="application/json",
    )


@app.get("/history/{conversation_id}")
async def history(conversation_id: str, limit: int = 10) -> Response:
    """
    GET /history/{conversation_id}?limit=10
    Returns the last `limit` turns for the conversation, oldest first.
    """
    limit = max(1, min(limit, 50))   # clamp to sane range
    try:
        from shared.conversation_store import async_get_history
        turns = await async_get_history(conversation_id, limit=limit)
    except Exception as exc:
        logger.error("History fetch failed: %s", exc, exc_info=True)
        turns = []

    return Response(
        content=json.dumps({
            "conversation_id": conversation_id,
            "turns":           turns,
            "count":           len(turns),
        }),
        media_type="application/json",
    )


@app.get("/memory/{user_id}")
async def get_memory(user_id: str) -> Response:
    """
    GET /memory/{user_id}
    Returns all long-term memory records for a user.
    """
    try:
        from shared.memory_store import get_long_term
        memories = await asyncio.to_thread(get_long_term, user_id)
    except Exception as exc:
        logger.error("Memory fetch failed: %s", exc, exc_info=True)
        memories = []

    return Response(
        content=json.dumps({
            "user_id":  user_id,
            "memories": memories,
            "count":    len(memories),
        }),
        media_type="application/json",
    )


if __name__ == "__main__":
    uvicorn.run("agents.main_agent:app", host="0.0.0.0", port=8000, reload=False)
