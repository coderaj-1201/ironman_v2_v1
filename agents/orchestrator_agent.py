"""
Orchestrator Agent — Local Dev
================================
MAF Functional Workflow (@workflow / @step).
Calls Retrieval Agent via direct HTTP (no Service Bus).
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
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from shared.azure_clients import get_openai_client
from shared.config import settings
from shared.logging_config import configure_logging, get_logger, set_correlation
from shared.models import (
    Domain, FinalResponse, OrchestratorRequest,
    RetrievalResult, RetrievalTool, UserQuery,
)

configure_logging(agent_name="orchestrator")
logger = get_logger(__name__)

_TOOL_LADDER   = [RetrievalTool.HYBRID, RetrievalTool.HYDE, RetrievalTool.DECOMPOSITION]
_RETRIEVAL_URL = "http://localhost:8002"

_CLASSIFY_SYSTEM_BASE = """Classify this enterprise query.
Return ONLY JSON: {"domain": "hr|legal|it", "tool": "hybrid|hyde|decomposition", "reason": "brief"}

domain: hr=people/leave/payroll/benefits, legal=contracts/compliance/GDPR/NDA, it=tech/infra/software/access
tool: hybrid=direct questions, hyde=vague/conceptual, decomposition=complex/multi-part"""


@step
async def classify_query(query: str, user_id: str = "") -> tuple[Domain, RetrievalTool]:
    system_prompt = _CLASSIFY_SYSTEM_BASE

    if user_id:
        try:
            from shared.memory_store import async_get_long_term_context
            lt_context = await async_get_long_term_context(user_id)
            if lt_context:
                system_prompt = f"{_CLASSIFY_SYSTEM_BASE}\n\n{lt_context}"
        except Exception as exc:
            logger.warning("classify_query: long-term memory fetch failed user_id=%s: %s", user_id, exc)

    try:
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
    except Exception as exc:
        logger.error("classify_query: OpenAI call failed — defaulting to it/hybrid: %s", exc, exc_info=True)
        return Domain.IT, RetrievalTool.HYBRID

    # Parse LLM response — guard against malformed JSON and invalid enum values
    try:
        raw    = json.loads(resp.choices[0].message.content)
        domain = Domain(raw.get("domain", "it"))
        tool   = RetrievalTool(raw.get("tool", "hybrid"))
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "classify_query: could not parse LLM response — defaulting to it/hybrid. "
            "raw='%.200s' error=%s",
            resp.choices[0].message.content if resp.choices else "empty", exc,
        )
        return Domain.IT, RetrievalTool.HYBRID

    logger.info(
        "classify domain=%s tool=%s reason='%s'",
        domain, tool, raw.get("reason", ""),
    )
    return domain, tool


@step
async def call_retrieval(req: OrchestratorRequest) -> RetrievalResult:
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{_RETRIEVAL_URL}/retrieve", json=req.__dict__)
            resp.raise_for_status()
            return RetrievalResult(**resp.json())
    except httpx.TimeoutException as exc:
        logger.error(
            "call_retrieval TIMEOUT attempt=%d tool=%s conversation_id=%s",
            req.attempt, req.tool, req.conversation_id,
        )
        raise
    except httpx.ConnectError as exc:
        logger.error(
            "call_retrieval CONNECTION REFUSED — retrieval agent down? "
            "attempt=%d conversation_id=%s",
            req.attempt, req.conversation_id,
        )
        raise
    except httpx.HTTPStatusError as exc:
        logger.error(
            "call_retrieval HTTP %s attempt=%d tool=%s body=%.200s",
            exc.response.status_code, req.attempt, req.tool, exc.response.text,
        )
        raise


@workflow(name="orchestrator_workflow")
async def orchestrator_workflow(user_query: UserQuery) -> FinalResponse:
    set_correlation(
        agent="orchestrator",
        conversation_id=user_query.conversation_id,
        question_id=user_query.question_id,
    )
    logger.info("orchestrator started query='%.80s'", user_query.text)

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
            question_id=user_query.question_id,
        )

        try:
            result      = await call_retrieval(req)
            last_result = result
        except httpx.TimeoutException:
            logger.warning(
                "orchestrator attempt=%d TIMED OUT — %s",
                attempt, "aborting retry loop" if attempt == settings.MAX_RETRIEVAL_ATTEMPTS else "retrying with next tool",
            )
            continue
        except httpx.ConnectError:
            # Retrieval agent is down — no point retrying
            logger.error("orchestrator aborting — retrieval agent unreachable attempt=%d", attempt)
            break
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code < 500:
                # 4xx won't recover with retries
                logger.error(
                    "orchestrator aborting — retrieval returned %s (non-retryable)",
                    exc.response.status_code,
                )
                break
            logger.warning("orchestrator attempt=%d retrieval 5xx — retrying", attempt)
            continue
        except Exception as exc:
            logger.error(
                "orchestrator attempt=%d unexpected retrieval error: %s",
                attempt, exc, exc_info=True,
            )
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
                question_id=user_query.question_id,
            )

        logger.warning(
            "orchestrator attempt=%d BELOW threshold — confidence=%.3f < %.2f tool=%s snippet='%.120s'",
            attempt, result.confidence, settings.CONFIDENCE_THRESHOLD,
            tool, result.answer.replace("\n", " "),
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
        question_id=user_query.question_id,
    )


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Orchestrator Agent started")
    yield
    logger.info("Orchestrator Agent stopped")


app = FastAPI(title="RAG Orchestrator Agent — Local", lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "unhandled exception path=%s: %s", request.url.path, exc, exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. The error has been logged."},
    )


@app.middleware("http")
async def correlation_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    set_correlation(request_id=request_id, agent="orchestrator")
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "agent": "orchestrator"}


@app.post("/orchestrate")
async def orchestrate(raw: Request) -> Response:
    try:
        body = await raw.json()
    except Exception as exc:
        logger.warning("orchestrate: invalid JSON body: %s", exc)
        return JSONResponse(status_code=400, content={"detail": "Invalid JSON body."})

    try:
        user_query = UserQuery(**body)
    except (TypeError, ValueError) as exc:
        logger.warning("orchestrate: invalid payload fields: %s body=%s", exc, body)
        return JSONResponse(status_code=422, content={"detail": f"Invalid payload: {exc}"})

    set_correlation(
        request_id=raw.headers.get("x-request-id", "-"),
        agent="orchestrator",
        conversation_id=user_query.conversation_id,
        question_id=user_query.question_id,
    )

    result_obj = await orchestrator_workflow.run(user_query)
    outputs    = result_obj.get_outputs()
    final: FinalResponse = outputs[0] if outputs else FinalResponse(
        status="failure", answer="", domain=None,
        conversation_id=user_query.conversation_id,
        user_id=user_query.user_id,
        question_id=user_query.question_id,
    )
    return Response(content=json.dumps(final.__dict__), media_type="application/json")


if __name__ == "__main__":
    uvicorn.run("agents.orchestrator_agent:app", host="0.0.0.0", port=8001, reload=False)
