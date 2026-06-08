"""
Orchestrator Agent — Local Dev
================================
MAF Functional Workflow (@workflow / @step).
Calls Retrieval Agent via direct HTTP (no Service Bus).

Changes vs original:
  - question_id propagated through OrchestratorRequest and FinalResponse.
  - classify_query: long-term memory context injected into system prompt when available.
  - All existing retry / tool-ladder logic is 100% unchanged.
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

import httpx
import uvicorn
from agent_framework import step, workflow
from fastapi import FastAPI, Request, Response

from shared.azure_clients import get_openai_client
from shared.config import settings
from shared.logging_config import configure_logging, get_logger
from shared.models import (
    Domain, FinalResponse, OrchestratorRequest,
    RetrievalResult, RetrievalTool, UserQuery,
)

configure_logging()
logger = get_logger(__name__)

_TOOL_LADDER   = [RetrievalTool.HYBRID, RetrievalTool.HYDE, RetrievalTool.DECOMPOSITION]
_RETRIEVAL_URL = "http://localhost:8002"

_CLASSIFY_SYSTEM_BASE = """Classify this enterprise query.
Return ONLY JSON: {"domain": "hr|legal|it", "tool": "hybrid|hyde|decomposition", "reason": "brief"}

domain: hr=people/leave/payroll/benefits, legal=contracts/compliance/GDPR/NDA, it=tech/infra/software/access
tool: hybrid=direct questions, hyde=vague/conceptual, decomposition=complex/multi-part"""


@step
async def classify_query(query: str, user_id: str = "") -> tuple[Domain, RetrievalTool]:
    """
    Classify domain + retrieval tool.
    If long-term memory is available for the user, it is appended to the system
    prompt to improve domain classification accuracy (e.g. a known HR user is
    less likely to be misclassified as IT).
    """
    system_prompt = _CLASSIFY_SYSTEM_BASE

    # Inject long-term memory context if available (non-fatal if it fails)
    if user_id:
        try:
            from shared.memory_store import async_get_long_term_context
            lt_context = await async_get_long_term_context(user_id)
            if lt_context:
                system_prompt = f"{_CLASSIFY_SYSTEM_BASE}\n\n{lt_context}"
        except Exception as exc:
            logger.warning("classify_query: could not fetch long-term memory: %s", exc)

    resp = await asyncio.to_thread(
        get_openai_client().chat.completions.create,
        model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Question: {query}"},
        ],
        temperature=0,
        max_tokens=120,
        response_format={"type": "json_object"},
    )
    raw    = json.loads(resp.choices[0].message.content)
    domain = Domain(raw.get("domain", "it"))
    tool   = RetrievalTool(raw.get("tool", "hybrid"))
    logger.info(
        "classify domain=%s tool=%s reason='%s'", domain, tool, raw.get("reason", "")
    )
    return domain, tool


@step
async def call_retrieval(req: OrchestratorRequest) -> RetrievalResult:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{_RETRIEVAL_URL}/retrieve", json=req.__dict__)
        resp.raise_for_status()
        return RetrievalResult(**resp.json())


@workflow(name="orchestrator_workflow")
async def orchestrator_workflow(user_query: UserQuery) -> FinalResponse:
    logger.info("orchestrator started query='%.80s'", user_query.text)

    # Pass user_id so classify_query can use long-term memory
    domain, _ = await classify_query(user_query.text, user_id=user_query.user_id)
    last_result: RetrievalResult | None = None

    for attempt_idx in range(settings.MAX_RETRIEVAL_ATTEMPTS):
        tool    = _TOOL_LADDER[attempt_idx]
        attempt = attempt_idx + 1

        logger.info(
            "orchestrator attempt=%d/%d domain=%s tool=%s",
            attempt, settings.MAX_RETRIEVAL_ATTEMPTS, domain, tool,
        )

        req = OrchestratorRequest(
            query=user_query.text,
            domain=domain,
            tool=tool,
            attempt=attempt,
            conversation_id=user_query.conversation_id,
            user_id=user_query.user_id,
            question_id=user_query.question_id,          # propagate
        )
        try:
            result      = await call_retrieval(req)
            last_result = result
        except Exception as exc:
            logger.error("Retrieval failed attempt=%d: %s", attempt, exc, exc_info=True)
            continue

        if result.passed:
            logger.info(
                "orchestrator SUCCESS attempt=%d confidence=%.3f", attempt, result.confidence
            )
            return FinalResponse(
                status="success",
                answer=result.answer,
                domain=domain,
                sources=result.sources,
                confidence=result.confidence,
                attempts_used=attempt,
                conversation_id=user_query.conversation_id,
                user_id=user_query.user_id,
                question_id=user_query.question_id,      # propagate
            )

        logger.warning(
            "orchestrator attempt=%d confidence=%.3f below threshold=%.2f",
            attempt, result.confidence, settings.CONFIDENCE_THRESHOLD,
        )

    logger.error("orchestrator FAILED after %d attempts", settings.MAX_RETRIEVAL_ATTEMPTS)
    return FinalResponse(
        status="failure",
        answer="",
        domain=domain,
        sources=last_result.sources    if last_result else [],
        confidence=last_result.confidence if last_result else 0.0,
        attempts_used=settings.MAX_RETRIEVAL_ATTEMPTS,
        conversation_id=user_query.conversation_id,
        user_id=user_query.user_id,
        question_id=user_query.question_id,              # propagate
    )


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Orchestrator Agent started.")
    yield
    logger.info("Orchestrator Agent stopped.")


app = FastAPI(title="RAG Orchestrator Agent — Local", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "agent": "orchestrator"}


@app.post("/orchestrate")
async def orchestrate(raw: Request) -> Response:
    body       = await raw.json()
    user_query = UserQuery(**body)
    result_obj = await orchestrator_workflow.run(user_query)
    outputs    = result_obj.get_outputs()
    final: FinalResponse = outputs[0] if outputs else FinalResponse(
        status="failure", answer="", domain=None,
        conversation_id=user_query.conversation_id,
        user_id=user_query.user_id,
        question_id=user_query.question_id,
    )
    return Response(
        content=json.dumps(final.__dict__),
        media_type="application/json",
    )


if __name__ == "__main__":
    uvicorn.run("agents.orchestrator_agent:app", host="0.0.0.0", port=8001, reload=False)
