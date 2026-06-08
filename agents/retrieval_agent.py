"""
Retrieval Agent — Local Dev
============================
MAF Functional Workflow (@workflow / @step).
Receives OrchestratorRequest via HTTP, returns RetrievalResult.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager

import uvicorn
from agent_framework import step, workflow
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from shared.azure_clients import get_openai_client
from shared.config import settings
from shared.logging_config import configure_logging, get_logger, set_correlation
from shared.models import OrchestratorRequest, RetrievalResult, SourceDocument
from tools.hybrid_search_tool import SearchDocument, fetch_parent_chunk, hybrid_search
from tools.hyde_tool import generate_hypothetical_document
from tools.query_decomposition_tool import decompose_query

configure_logging(agent_name="retrieval")
logger = get_logger(__name__)


# ── Retrieval steps ───────────────────────────────────────────────────────────

@step
async def run_hybrid(query: str, domain: str) -> list[SearchDocument]:
    try:
        return await asyncio.to_thread(hybrid_search, query, domain)
    except Exception as exc:
        logger.error("run_hybrid FAILED domain=%s: %s", domain, exc, exc_info=True)
        return []


@step
async def run_hyde(query: str, domain: str) -> list[SearchDocument]:
    try:
        hypo = await asyncio.to_thread(generate_hypothetical_document, query)
    except Exception as exc:
        logger.warning(
            "run_hyde: hypothetical doc generation failed — falling back to direct query: %s", exc
        )
        hypo = query  # graceful fallback: search with original query
    try:
        return await asyncio.to_thread(hybrid_search, hypo, domain)
    except Exception as exc:
        logger.error("run_hyde hybrid_search FAILED domain=%s: %s", domain, exc, exc_info=True)
        return []


@step
async def run_decomposition(query: str, domain: str) -> list[SearchDocument]:
    try:
        sub_queries = await asyncio.to_thread(decompose_query, query)
    except Exception as exc:
        logger.warning(
            "run_decomposition: decompose_query failed — falling back to original query: %s", exc
        )
        sub_queries = [query]

    try:
        result_sets = await asyncio.gather(
            *[asyncio.to_thread(hybrid_search, sq, domain) for sq in sub_queries],
            return_exceptions=True,
        )
    except Exception as exc:
        logger.error("run_decomposition: gather failed domain=%s: %s", domain, exc, exc_info=True)
        return []

    seen: dict[str, SearchDocument] = {}
    for i, docs_or_exc in enumerate(result_sets):
        if isinstance(docs_or_exc, Exception):
            logger.warning(
                "run_decomposition: sub_query[%d] '%s' failed: %s",
                i, sub_queries[i] if i < len(sub_queries) else "?", docs_or_exc,
            )
            continue
        for doc in docs_or_exc:
            if doc.id not in seen or doc.score > seen[doc.id].score:
                seen[doc.id] = doc

    return sorted(seen.values(), key=lambda d: d.score, reverse=True)[: settings.RETRIEVAL_TOP_K]


_SYNTHESIS_SYSTEM = """You are an enterprise knowledge assistant.
Answer using ONLY the context below. Be concise and cite source names.

After your answer output ONLY this JSON on a new line (no markdown):
{"confidence": <0.0-1.0>}

Confidence: 0.9+ = fully answers, 0.7-0.89 = mostly, 0.5-0.69 = partial, <0.5 = insufficient."""


@step
async def synthesize_answer(
    query: str,
    all_docs: list[SearchDocument],
    conversation_id: str = "",
    question_id: str = "",
) -> tuple[str, float, list[SourceDocument]]:
    if not all_docs:
        logger.warning(
            "synthesize_answer: no docs passed — returning zero-confidence answer "
            "conversation_id=%s question_id=%s",
            conversation_id, question_id,
        )
        return "No relevant information found in the knowledge base.", 0.0, []

    context_parts = []
    for i, d in enumerate(all_docs):
        heading = getattr(d, "section_heading", "")
        page    = getattr(d, "page_number", 0)
        label   = (
            f"[{i+1}] Source: {d.source}"
            + (f" (p.{page})" if page else "")
            + (f" | {heading}" if heading else "")
        )
        if getattr(d, "chunk_type", "") == "table" and getattr(d, "table_raw", ""):
            context_parts.append(f"{label}\nSummary: {d.content}\nTable:\n{d.table_raw}")
        else:
            context_parts.append(f"{label}\n{d.content}")
    context = "\n\n".join(context_parts)

    user_content = f"Context:\n{context}\n\nQuestion: {query}"
    if conversation_id:
        try:
            from shared.memory_store import async_get_short_term_context
            st_context = await async_get_short_term_context(conversation_id)
            if st_context:
                user_content = f"{st_context}\n\n{user_content}"
        except Exception as exc:
            logger.warning(
                "synthesize_answer: short-term memory fetch failed conversation_id=%s: %s",
                conversation_id, exc,
            )

    try:
        resp = await asyncio.wait_for(
            asyncio.to_thread(
                get_openai_client().chat.completions.create,
                model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": _SYNTHESIS_SYSTEM},
                    {"role": "user",   "content": user_content},
                ],
                temperature=settings.SYNTHESIS_TEMPERATURE,
                max_tokens=800,
            ),
            timeout=45.0,  # explicit timeout — don't let synthesis block indefinitely
        )
    except asyncio.TimeoutError:
        logger.error(
            "synthesize_answer: OpenAI call TIMED OUT (45s) "
            "conversation_id=%s question_id=%s",
            conversation_id, question_id,
        )
        return "Answer synthesis timed out. Please try again.", 0.0, []
    except Exception as exc:
        logger.error(
            "synthesize_answer: OpenAI call FAILED conversation_id=%s: %s",
            conversation_id, exc, exc_info=True,
        )
        return "Answer synthesis failed due to an internal error.", 0.0, []

    full_text = resp.choices[0].message.content.strip()
    logger.debug(
        "synthesize_answer raw LLM output conversation_id=%s: '%.200s'",
        conversation_id, full_text.replace("\n", "\\n"),
    )

    try:
        split_idx = full_text.rfind("\n{")
        if split_idx == -1:
            split_idx = full_text.rfind('{"confidence"')
        answer     = full_text[:split_idx].strip() if split_idx > 0 else full_text
        confidence = float(json.loads(full_text[split_idx:]).get("confidence", 0.0))
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "synthesize_answer: could not parse confidence JSON — defaulting to 0.5. "
            "conversation_id=%s raw_suffix='%.100s' error=%s",
            conversation_id,
            full_text[max(0, len(full_text) - 100):],
            exc,
        )
        answer, confidence = full_text, 0.5

    sources = [
        SourceDocument(title=d.source, excerpt=d.content[:200], url="", relevance=round(d.score, 3))
        for d in all_docs[:3]
    ]
    return answer, round(min(max(confidence, 0.0), 1.0), 3), sources


# ── Main retrieval workflow ───────────────────────────────────────────────────

@workflow(name="retrieval_workflow")
async def retrieval_workflow(request: OrchestratorRequest) -> RetrievalResult:
    set_correlation(
        agent="retrieval",
        conversation_id=request.conversation_id,
        question_id=request.question_id,
    )
    logger.info(
        "retrieval attempt=%d domain=%s tool=%s query='%.80s'",
        request.attempt, request.domain, request.tool, request.query,
    )

    match request.tool:
        case "hyde":          docs = await run_hyde(request.query, request.domain)
        case "decomposition": docs = await run_decomposition(request.query, request.domain)
        case _:               docs = await run_hybrid(request.query, request.domain)

    if not docs:
        logger.warning(
            "retrieval attempt=%d tool=%s returned ZERO chunks domain=%s query='%.80s'",
            request.attempt, request.tool, request.domain, request.query,
        )
    else:
        logger.debug(
            "retrieval attempt=%d tool=%s pulled %d chunks top_score=%.3f",
            request.attempt, request.tool, len(docs), docs[0].score,
        )

    # Parent-child context enrichment
    parent_ids  = list({d.parent_id for d in docs if d.parent_id})
    parent_docs = []
    for pid in parent_ids[:3]:
        try:
            parent = await asyncio.to_thread(fetch_parent_chunk, pid)
            if parent:
                parent_docs.append(parent)
        except Exception as exc:
            logger.warning("fetch_parent_chunk FAILED pid=%s: %s", pid, exc)

    if parent_docs:
        logger.debug(
            "retrieval attempt=%d parent enrichment: %d parents for %d child chunks",
            request.attempt, len(parent_docs), len(docs),
        )

    all_docs = docs + [p for p in parent_docs if p.id not in {d.id for d in docs}]

    answer, confidence, source_docs = await synthesize_answer(
        request.query, all_docs,
        conversation_id=request.conversation_id,
        question_id=request.question_id,
    )

    logger.info(
        "retrieval complete attempt=%d confidence=%.3f passed=%s chunks_used=%d",
        request.attempt, confidence,
        confidence >= settings.CONFIDENCE_THRESHOLD,
        len(all_docs),
    )

    return RetrievalResult(
        query=request.query,
        domain=request.domain,
        tool=request.tool,
        attempt=request.attempt,
        answer=answer,
        confidence=confidence,
        sources=[
            {"title": s.title, "excerpt": s.excerpt, "url": s.url, "relevance": s.relevance}
            for s in source_docs
        ],
        conversation_id=request.conversation_id,
        user_id=request.user_id,
        question_id=request.question_id,
    )


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Retrieval Agent started")
    yield
    logger.info("Retrieval Agent stopped")


app = FastAPI(title="RAG Retrieval Agent — Local", lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled exception path=%s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. The error has been logged."},
    )


@app.middleware("http")
async def correlation_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    set_correlation(request_id=request_id, agent="retrieval")
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "agent": "retrieval"}


@app.post("/retrieve")
async def retrieve(raw: Request) -> Response:
    try:
        body = await raw.json()
    except Exception as exc:
        logger.warning("retrieve: invalid JSON body: %s", exc)
        return JSONResponse(status_code=400, content={"detail": "Invalid JSON body."})

    try:
        request = OrchestratorRequest(**body)
    except (TypeError, ValueError) as exc:
        logger.warning("retrieve: invalid payload: %s body=%s", exc, body)
        return JSONResponse(status_code=422, content={"detail": f"Invalid payload: {exc}"})

    set_correlation(
        request_id=raw.headers.get("x-request-id", "-"),
        agent="retrieval",
        conversation_id=request.conversation_id,
        question_id=request.question_id,
    )

    result_obj = await retrieval_workflow.run(request)
    outputs    = result_obj.get_outputs()
    result: RetrievalResult = outputs[0] if outputs else RetrievalResult(
        query=request.query, domain=request.domain, tool=request.tool,
        attempt=request.attempt, answer="Internal error.", confidence=0.0,
        sources=[], conversation_id=request.conversation_id,
        user_id=request.user_id, question_id=request.question_id,
    )
    return Response(content=json.dumps(result.__dict__), media_type="application/json")


if __name__ == "__main__":
    uvicorn.run("agents.retrieval_agent:app", host="0.0.0.0", port=8002, reload=False)
