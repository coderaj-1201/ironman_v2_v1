"""
Main Agent — Local Dev
=======================
MAF Functional Workflow (@workflow / @step).
Entry point for all queries. Calls Orchestrator via HTTP.

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
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

import httpx
import uvicorn
from agent_framework import step, workflow
from azure.cosmos.exceptions import CosmosHttpResponseError
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from shared.azure_clients import get_chat_history_container, get_feedback_container
from shared.config import settings
from shared.logging_config import configure_logging, get_logger, set_correlation
from shared.models import FinalResponse, UserQuery

configure_logging(agent_name="main")
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


# ── Pydantic request/response models ─────────────────────────────────────────

class QueryRequest(BaseModel):
    text: str
    conversation_id: str = ""
    user_id: str = "anonymous"

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


# ── Cosmos helpers ────────────────────────────────────────────────────────────

def _upsert_chat_turn(doc: dict) -> None:
    try:
        get_chat_history_container().upsert_item(doc)
        logger.info(
            "cosmos chat_history saved question_id=%s answer_id=%s",
            doc["question_id"], doc["answer_id"],
        )
    except CosmosHttpResponseError as exc:
        logger.error(
            "cosmos chat_history write FAILED status=%s question_id=%s: %s",
            exc.status_code, doc.get("question_id"), exc, exc_info=True,
        )
        # Non-fatal: don't propagate — history loss is preferable to failed responses


def _cosmos_write_callback(task: asyncio.Task) -> None:
    """Attached as done_callback on fire-and-forget Cosmos tasks."""
    exc = task.exception()
    if exc:
        logger.error("cosmos background write raised unhandled exception: %s", exc, exc_info=exc)


def _upsert_feedback(doc: dict) -> None:
    try:
        get_feedback_container().upsert_item(doc)
        logger.info(
            "cosmos feedback saved feedback_id=%s answer_id=%s",
            doc["id"], doc["answer_id"],
        )
    except CosmosHttpResponseError as exc:
        logger.error(
            "cosmos feedback write FAILED status=%s answer_id=%s: %s",
            exc.status_code, doc.get("answer_id"), exc, exc_info=True,
        )
        raise  # feedback write IS fatal — caller returns 502


def _query_chat_history(conversation_id: str, offset: int, limit: int) -> tuple[list[dict], int]:
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
    except CosmosHttpResponseError as exc:
        logger.warning(
            "cosmos feedback read FAILED status=%s answer_id=%s: %s",
            exc.status_code, answer_id, exc,
        )
        return None


# ── Meta extraction (fallback only) ──────────────────────────────────────────

def _parse_meta_from_reply(reply: str) -> tuple[str | None, float, int, str]:
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
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{_ORCHESTRATOR_URL}/orchestrate",
                json=user_query.__dict__,
            )
            resp.raise_for_status()
            return FinalResponse(**resp.json())
    except httpx.TimeoutException as exc:
        logger.error(
            "call_orchestrator TIMEOUT conversation_id=%s question_id=%s: %s",
            user_query.conversation_id, user_query.question_id, exc,
        )
        raise
    except httpx.ConnectError as exc:
        logger.error(
            "call_orchestrator CONNECTION REFUSED — orchestrator down? "
            "conversation_id=%s: %s",
            user_query.conversation_id, exc,
        )
        raise
    except httpx.HTTPStatusError as exc:
        logger.error(
            "call_orchestrator HTTP %s conversation_id=%s body=%.200s",
            exc.response.status_code, user_query.conversation_id,
            exc.response.text,
        )
        raise


@step
async def handle_raise_ticket(user_id: str) -> str:
    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    logger.info("ticket raised ticket_id=%s user_id=%s", ticket_id, user_id)
    return (f"✅ **Ticket raised!** Reference: `{ticket_id}`\n"
            f"Expected response: **4 business hours**.")


@step
async def handle_connect_sme(user_id: str) -> str:
    logger.info("sme connect requested user_id=%s", user_id)
    return "✅ **Connecting you with an SME.** Expected response: **2 business hours**."


@workflow(name="main_agent_workflow")
async def main_agent_workflow(user_query: UserQuery) -> tuple[str, FinalResponse | None]:
    """
    Returns (reply_string, FinalResponse | None).
    reply   — formatted display string for Teams / chat UI.
    final   — structured data for API response; None for ticket/sme flows.
    """
    text = user_query.text.strip().lower()

    if text == "raise_ticket":
        return await handle_raise_ticket(user_query.user_id), None
    if text == "connect_sme":
        return await handle_connect_sme(user_query.user_id), None

    try:
        final: FinalResponse = await call_orchestrator(user_query)
    except httpx.TimeoutException:
        logger.error(
            "main_agent_workflow: orchestrator timed out conversation_id=%s",
            user_query.conversation_id,
        )
        return _FAILURE_MSG, None
    except (httpx.ConnectError, httpx.HTTPStatusError):
        return _FAILURE_MSG, None
    except Exception as exc:
        logger.error(
            "main_agent_workflow: unexpected error conversation_id=%s: %s",
            user_query.conversation_id, exc, exc_info=True,
        )
        return _FAILURE_MSG, None

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
        return f"{final.answer}{sources_text}\n\n{meta}", final

    return _FAILURE_MSG, final


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Main Agent started")
    yield
    logger.info("Main Agent stopped")


app = FastAPI(title="RAG Main Agent — Local", lifespan=lifespan)


# ── Global exception handler — prevents raw 500s leaking stack traces ─────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "unhandled exception path=%s method=%s: %s",
        request.url.path, request.method, exc, exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. The error has been logged."},
    )


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError) -> JSONResponse:
    logger.warning("request validation error path=%s: %s", request.url.path, exc)
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


# ── Middleware: inject request_id + bind correlation on every request ──────────

@app.middleware("http")
async def correlation_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    # Conversation/question IDs not yet known — set after body parse in endpoint
    set_correlation(request_id=request_id, agent="main")
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "agent": "main"}


@app.post("/query")
async def query(req: QueryRequest, request: Request) -> Response:
    conversation_id = req.conversation_id or str(uuid.uuid4())
    question_id     = new_question_id()
    answer_id       = new_answer_id()
    request_id      = request.headers.get("x-request-id", str(uuid.uuid4()))

    # Bind full correlation context now that IDs are known
    set_correlation(
        request_id=request_id,
        agent="main",
        conversation_id=conversation_id,
        question_id=question_id,
    )
    logger.info(
        "query received user_id=%s text_len=%d",
        req.user_id, len(req.text),
    )

    user_query = UserQuery(
        text=req.text,
        conversation_id=conversation_id,
        user_id=req.user_id,
        question_id=question_id,
    )

    result_obj = await main_agent_workflow.run(user_query)
    outputs    = result_obj.get_outputs()

    if outputs and isinstance(outputs[0], tuple):
        reply, final = outputs[0]
    else:
        reply, final = _FAILURE_MSG, None

    # Derive structured fields
    text_lower = req.text.strip().lower()
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
        answer        = final.answer
    else:
        domain, confidence, attempts_used, status = _parse_meta_from_reply(reply)
        sources, answer = [], reply

    logger.info(
        "query complete status=%s domain=%s confidence=%.3f attempts=%d answer_id=%s",
        status, domain, confidence, attempts_used, answer_id,
    )

    # Persist to Cosmos — fire-and-forget with error callback
    turn_doc = {
        "id":              question_id,
        "question_id":     question_id,
        "answer_id":       answer_id,
        "conversation_id": conversation_id,
        "user_id":         req.user_id,
        "question":        req.text,
        "answer":          answer,
        "domain":          domain,
        "confidence":      confidence,
        "attempts_used":   attempts_used,
        "status":          status,
        "created_at":      _utc_now(),
    }
    task = asyncio.create_task(asyncio.to_thread(_upsert_chat_turn, turn_doc))
    task.add_done_callback(_cosmos_write_callback)

    return Response(
        content=json.dumps({
            "reply":           reply,
            "question_id":     question_id,
            "answer_id":       answer_id,
            "conversation_id": conversation_id,
            "answer":          answer,
            "domain":          domain,
            "confidence":      confidence,
            "attempts_used":   attempts_used,
            "status":          status,
            "sources":         sources,
        }),
        media_type="application/json",
    )


@app.get("/history/{conversation_id}", response_model=HistoryResponse)
async def get_history(
    conversation_id: str,
    page: int      = Query(default=1,  ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    set_correlation(agent="main", conversation_id=conversation_id)
    offset = (page - 1) * page_size
    try:
        turns, total = await asyncio.to_thread(
            _query_chat_history, conversation_id, offset, page_size
        )
    except CosmosHttpResponseError as exc:
        logger.error(
            "get_history cosmos read FAILED status=%s conversation_id=%s: %s",
            exc.status_code, conversation_id, exc,
        )
        raise HTTPException(status_code=502, detail="Could not read chat history from Cosmos DB.")

    if not turns and page == 1:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    return HistoryResponse(
        conversation_id=conversation_id,
        total=total, page=page, page_size=page_size, turns=turns,
    )


@app.post("/feedback", response_model=FeedbackResponse, status_code=201)
async def submit_feedback(req: FeedbackRequest, request: Request):
    set_correlation(
        request_id=request.headers.get("x-request-id", "-"),
        agent="main",
        question_id=req.question_id,
    )
    logger.info(
        "feedback received answer_id=%s rating=%s user_id=%s",
        req.answer_id, req.rating, req.user_id,
    )

    existing = await asyncio.to_thread(_get_feedback_by_answer, req.answer_id)
    if existing:
        logger.warning("feedback duplicate answer_id=%s user_id=%s", req.answer_id, req.user_id)
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
        raise HTTPException(status_code=502, detail="Could not save feedback to Cosmos DB.")

    return FeedbackResponse(**doc)


@app.get("/feedback/{answer_id}", response_model=FeedbackResponse)
async def get_feedback(answer_id: str):
    set_correlation(agent="main")
    try:
        doc = await asyncio.to_thread(_get_feedback_by_answer, answer_id)
    except CosmosHttpResponseError as exc:
        logger.error(
            "get_feedback cosmos read FAILED status=%s answer_id=%s: %s",
            exc.status_code, answer_id, exc,
        )
        raise HTTPException(status_code=502, detail="Could not read feedback from Cosmos DB.")

    if not doc:
        raise HTTPException(status_code=404, detail=f"No feedback for answer_id={answer_id}.")

    return FeedbackResponse(**doc)


if __name__ == "__main__":
    uvicorn.run("agents.main_agent:app", host="0.0.0.0", port=8000, reload=False)
