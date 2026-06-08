"""
Main Agent — Local Dev
=======================
MAF Functional Workflow (@workflow / @step).
Entry point for all queries. Calls Orchestrator via HTTP.

Chat history and feedback are persisted to Azure Cosmos DB.

ID scheme
---------
  question_id : qst-<8 hex>   — unique per user message, stored on the turn
  answer_id   : ans-<8 hex>   — unique per synthesised answer, links feedback → turn

Cosmos containers
-----------------
  chat_history  (partition key: /conversation_id)
      id                = question_id          ← Cosmos document id
      question_id       = same as id
      answer_id         = ans-<hex>
      conversation_id   = ...
      user_id           = ...
      question          = user text
      answer            = agent reply
      domain            = hr | legal | it | null
      confidence        = float
      attempts_used     = int
      status            = success | failure
      created_at        = ISO-8601 UTC

  feedback  (partition key: /answer_id)
      id          = feedback_id               ← Cosmos document id
      feedback_id = same as id
      answer_id   = ans-<hex>
      question_id = qst-<hex>                 ← back-link to the turn
      conversation_id, user_id, rating, comment, created_at

Endpoints
---------
  POST /query
  GET  /history/{conversation_id}?page=1&page_size=20
  POST /feedback
  GET  /feedback/{answer_id}
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

import httpx
import uvicorn
from agent_framework import step, workflow
from azure.cosmos.exceptions import CosmosHttpResponseError
from fastapi import FastAPI, HTTPException, Query, Request, Response
from pydantic import BaseModel

from shared.azure_clients import get_chat_history_container, get_feedback_container
from shared.config import settings
from shared.logging_config import configure_logging, get_logger
from shared.models import FinalResponse, UserQuery

configure_logging()
logger = get_logger(__name__)

_ORCHESTRATOR_URL = "http://localhost:8001"

_FAILURE_MSG = (
    "I wasn't able to find a confident answer after exhausting all retrieval strategies.\n\n"
    "📋 **Option 1 — Raise a Support Ticket**\nReply with: `raise_ticket`\n\n"
    "👤 **Option 2 — Connect with a Subject Matter Expert**\nReply with: `connect_sme`"
)


# ── ID generators ─────────────────────────────────────────────────────────────

def new_question_id() -> str:
    return f"qst-{uuid.uuid4().hex[:8]}"

def new_answer_id() -> str:
    return f"ans-{uuid.uuid4().hex[:8]}"

def new_feedback_id() -> str:
    return f"fb-{uuid.uuid4().hex[:10]}"

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Pydantic API models ───────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    answer_id: str
    question_id: str
    conversation_id: str
    user_id: str
    rating: Literal["up", "down"]
    comment: str | None = None

class FeedbackResponse(BaseModel):
    feedback_id: str
    answer_id: str
    question_id: str
    conversation_id: str
    user_id: str
    rating: str
    comment: str | None
    created_at: str

class HistoryResponse(BaseModel):
    conversation_id: str
    total: int
    page: int
    page_size: int
    turns: list[dict]


# ── Cosmos helpers (run in thread — SDK is sync) ──────────────────────────────

def _upsert_chat_turn(doc: dict) -> None:
    try:
        get_chat_history_container().upsert_item(doc)
        logger.info(
            "cosmos chat_history saved question_id=%s answer_id=%s",
            doc["question_id"], doc["answer_id"],
        )
    except CosmosHttpResponseError as exc:
        logger.error("cosmos chat_history write failed: %s", exc, exc_info=True)
        raise


def _upsert_feedback(doc: dict) -> None:
    try:
        get_feedback_container().upsert_item(doc)
        logger.info("cosmos feedback saved feedback_id=%s answer_id=%s", doc["id"], doc["answer_id"])
    except CosmosHttpResponseError as exc:
        logger.error("cosmos feedback write failed: %s", exc, exc_info=True)
        raise


def _query_chat_history(conversation_id: str, offset: int, limit: int) -> tuple[list[dict], int]:
    """Returns (page_of_turns, total_count). Ordered by _ts ASC (insertion order)."""
    container = get_chat_history_container()

    total = list(container.query_items(
        query="SELECT VALUE COUNT(1) FROM c WHERE c.conversation_id = @cid",
        parameters=[{"name": "@cid", "value": conversation_id}],
        partition_key=conversation_id,
    ))
    total_count: int = total[0] if total else 0

    items = list(container.query_items(
        query=(
            "SELECT * FROM c WHERE c.conversation_id = @cid "
            "ORDER BY c._ts ASC OFFSET @offset LIMIT @limit"
        ),
        parameters=[
            {"name": "@cid",    "value": conversation_id},
            {"name": "@offset", "value": offset},
            {"name": "@limit",  "value": limit},
        ],
        partition_key=conversation_id,
    ))
    return items, total_count


def _get_feedback_by_answer(answer_id: str) -> dict | None:
    try:
        items = list(get_feedback_container().query_items(
            query="SELECT * FROM c WHERE c.answer_id = @aid",
            parameters=[{"name": "@aid", "value": answer_id}],
            partition_key=answer_id,
        ))
        return items[0] if items else None
    except CosmosHttpResponseError:
        return None


# ── Meta extraction from formatted reply ─────────────────────────────────────

def _parse_meta_from_reply(reply: str) -> tuple[str | None, float, int, str]:
    """Extract (domain, confidence, attempts, status) from the footer injected by workflow."""
    domain: str | None = None
    confidence: float = 0.0
    attempts: int = 0
    status = "failure"

    m_domain = re.search(r"Domain:\s*(\w+)", reply)
    m_conf   = re.search(r"Confidence:\s*(\d+)%", reply)
    m_att    = re.search(r"Attempts:\s*(\d+)", reply)

    if m_domain:
        domain = m_domain.group(1).lower()
        status = "success"
    if m_conf:
        confidence = int(m_conf.group(1)) / 100.0
    if m_att:
        attempts = int(m_att.group(1))

    return domain, confidence, attempts, status


# ── MAF steps / workflow ──────────────────────────────────────────────────────

@step
async def call_orchestrator(user_query: UserQuery) -> FinalResponse:
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{_ORCHESTRATOR_URL}/orchestrate", json=user_query.__dict__)
        resp.raise_for_status()
        return FinalResponse(**resp.json())


@step
async def handle_raise_ticket(user_id: str) -> str:
    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    logger.info("Ticket raised ticket_id=%s user_id=%s", ticket_id, user_id)
    return (f"✅ **Ticket raised!** Reference: `{ticket_id}`\n"
            f"Expected response: **4 business hours**.")


@step
async def handle_connect_sme(user_id: str) -> str:
    logger.info("SME connect requested user_id=%s", user_id)
    return "✅ **Connecting you with an SME.** Expected response: **2 business hours**."


@workflow(name="main_agent_workflow")
async def main_agent_workflow(user_query: UserQuery) -> tuple[str, FinalResponse | None]:
    """
    Returns (reply_string, FinalResponse | None).
    reply_string  — formatted text for display (Teams / chat UI).
    FinalResponse — structured data for API consumers; None for ticket/sme flows.
    """
    text = user_query.text.strip().lower()

    if text == "raise_ticket":
        return await handle_raise_ticket(user_query.user_id), None
    if text == "connect_sme":
        return await handle_connect_sme(user_query.user_id), None

    try:
        final: FinalResponse = await call_orchestrator(user_query)
    except Exception as exc:
        logger.error("Orchestrator call failed: %s", exc, exc_info=True)
        return _FAILURE_MSG, None

    if final.status == "success":
        sources_text = ""
        if final.sources:
            bullets = "\n".join(f"  • {s.get('title', s.get('source', ''))}" for s in final.sources)
            sources_text = f"\n\n📚 **Sources:**\n{bullets}"
        meta = (f"*Domain: {final.domain.upper() if final.domain else 'N/A'} | "
                f"Confidence: {final.confidence:.0%} | Attempts: {final.attempts_used}*")
        reply = f"{final.answer}{sources_text}\n\n{meta}"
        return reply, final

    return _FAILURE_MSG, final


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
    )

    # Generate IDs before workflow runs
    question_id = new_question_id()
    answer_id   = new_answer_id()

    result_obj = await main_agent_workflow.run(user_query)
    outputs    = result_obj.get_outputs()

    # Unpack (reply, FinalResponse | None) tuple from refactored workflow
    if outputs and isinstance(outputs[0], tuple):
        reply, final = outputs[0]
    else:
        reply, final = _FAILURE_MSG, None

    # Derive structured fields — use FinalResponse directly, no regex needed
    text_lower = user_query.text.strip().lower()
    if text_lower in ("raise_ticket", "connect_sme"):
        domain, confidence, attempts_used, status, sources, answer = (
            None, 0.0, 0, "success", [], reply
        )
    elif final is not None:
        domain        = final.domain
        confidence    = final.confidence
        attempts_used = final.attempts_used
        status        = final.status
        sources       = final.sources
        answer        = final.answer       # clean answer text without footer
    else:
        # orchestrator unreachable — regex fallback
        domain, confidence, attempts_used, status = _parse_meta_from_reply(reply)
        sources, answer = [], reply

    # Persist to Cosmos (fire-and-forget — response not blocked)
    turn_doc = {
        "id":              question_id,   # Cosmos requires 'id'
        "question_id":     question_id,
        "answer_id":       answer_id,
        "conversation_id": user_query.conversation_id,
        "user_id":         user_query.user_id,
        "question":        user_query.text,
        "answer":          answer,
        "domain":          domain,
        "confidence":      confidence,
        "attempts_used":   attempts_used,
        "status":          status,
        "created_at":      _utc_now(),
    }
    asyncio.create_task(asyncio.to_thread(_upsert_chat_turn, turn_doc))

    return Response(
        content=json.dumps({
            # ── Display ───────────────────────────────────────────────────────
            "reply":          reply,          # full formatted string for UI / Teams
            # ── Identity ─────────────────────────────────────────────────────
            "question_id":    question_id,
            "answer_id":      answer_id,
            "conversation_id": user_query.conversation_id,
            # ── Structured fields ─────────────────────────────────────────────
            "answer":         answer,         # clean answer text, no footer
            "domain":         domain,
            "confidence":     confidence,
            "attempts_used":  attempts_used,
            "status":         status,
            "sources":        sources,        # list of {title, excerpt, url, relevance}
        }),
        media_type="application/json",
    )


# ── Chat history endpoint ─────────────────────────────────────────────────────

@app.get("/history/{conversation_id}", response_model=HistoryResponse)
async def get_history(
    conversation_id: str,
    page: int      = Query(default=1,  ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    offset = (page - 1) * page_size
    try:
        turns, total = await asyncio.to_thread(
            _query_chat_history, conversation_id, offset, page_size
        )
    except CosmosHttpResponseError as exc:
        logger.error("cosmos history read failed: %s", exc)
        raise HTTPException(status_code=502, detail="Could not read chat history from Cosmos DB.")

    if not turns and page == 1:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    return HistoryResponse(
        conversation_id=conversation_id,
        total=total,
        page=page,
        page_size=page_size,
        turns=turns,
    )


# ── Feedback endpoints ────────────────────────────────────────────────────────

@app.post("/feedback", response_model=FeedbackResponse, status_code=201)
async def submit_feedback(req: FeedbackRequest):
    existing = await asyncio.to_thread(_get_feedback_by_answer, req.answer_id)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Feedback already submitted for answer_id={req.answer_id}.",
        )

    feedback_id = new_feedback_id()
    doc = {
        "id":              feedback_id,
        "feedback_id":     feedback_id,
        "answer_id":       req.answer_id,
        "question_id":     req.question_id,
        "conversation_id": req.conversation_id,
        "user_id":         req.user_id,
        "rating":          req.rating,
        "comment":         req.comment,
        "created_at":      _utc_now(),
    }

    try:
        await asyncio.to_thread(_upsert_feedback, doc)
    except CosmosHttpResponseError as exc:
        logger.error("cosmos feedback write error: %s", exc)
        raise HTTPException(status_code=502, detail="Could not save feedback to Cosmos DB.")

    return FeedbackResponse(**doc)


@app.get("/feedback/{answer_id}", response_model=FeedbackResponse)
async def get_feedback(answer_id: str):
    try:
        doc = await asyncio.to_thread(_get_feedback_by_answer, answer_id)
    except CosmosHttpResponseError as exc:
        logger.error("cosmos feedback read error: %s", exc)
        raise HTTPException(status_code=502, detail="Could not read feedback from Cosmos DB.")

    if not doc:
        raise HTTPException(status_code=404, detail=f"No feedback for answer_id={answer_id}.")

    return FeedbackResponse(**doc)


if __name__ == "__main__":
    uvicorn.run("agents.main_agent:app", host="0.0.0.0", port=8000, reload=False)
