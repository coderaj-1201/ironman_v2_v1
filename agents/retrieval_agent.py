"""
Retrieval Agent — Local Dev
============================
MAF Functional Workflow (@workflow / @step).
Receives OrchestratorRequest via HTTP, returns RetrievalResult.
No Service Bus — orchestrator calls /retrieve directly.

Changes vs original:
  - question_id propagated through RetrievalResult.
  - synthesize_answer: short-term conversation history injected as context
    (prepended to the user message, not the system prompt — keeps system prompt clean).
  - All retrieval / parent-child logic is 100% unchanged.
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

import uvicorn
from agent_framework import step, workflow
from fastapi import FastAPI, Request, Response

from shared.azure_clients import get_openai_client
from shared.config import settings
from shared.logging_config import configure_logging, get_logger
from shared.models import OrchestratorRequest, RetrievalResult, SourceDocument
from tools.hybrid_search_tool import SearchDocument, fetch_parent_chunk, hybrid_search
from tools.hyde_tool import generate_hypothetical_document
from tools.query_decomposition_tool import decompose_query

configure_logging()
logger = get_logger(__name__)


# ── Retrieval steps — unchanged from original ─────────────────────────────────

@step
async def run_hybrid(query: str, domain: str) -> list[SearchDocument]:
    return await asyncio.to_thread(hybrid_search, query, domain)


@step
async def run_hyde(query: str, domain: str) -> list[SearchDocument]:
    hypo = await asyncio.to_thread(generate_hypothetical_document, query)
    return await asyncio.to_thread(hybrid_search, hypo, domain)


@step
async def run_decomposition(query: str, domain: str) -> list[SearchDocument]:
    sub_queries = await asyncio.to_thread(decompose_query, query)
    result_sets = await asyncio.gather(
        *[asyncio.to_thread(hybrid_search, sq, domain) for sq in sub_queries]
    )
    seen: dict[str, SearchDocument] = {}
    for docs in result_sets:
        for doc in docs:
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
) -> tuple[str, float, list[SourceDocument]]:
    """
    Synthesize an answer from retrieved documents.
    If conversation_id is provided, the last N turns are prepended to the
    user message so the LLM can resolve follow-up questions (e.g. "what about
    part-time employees?" after asking about leave policy).
    """
    if not all_docs:
        return "No relevant information found in the knowledge base.", 0.0, []

    # Build context block — unchanged from original
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

    # Build user message — prepend short-term history if available
    user_content = f"Context:\n{context}\n\nQuestion: {query}"
    if conversation_id:
        try:
            from shared.memory_store import async_get_short_term_context
            st_context = await async_get_short_term_context(conversation_id)
            if st_context:
                user_content = f"{st_context}\n\n{user_content}"
        except Exception as exc:
            logger.warning("synthesize_answer: short-term memory fetch failed: %s", exc)

    resp = await asyncio.to_thread(
        get_openai_client().chat.completions.create,
        model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": _SYNTHESIS_SYSTEM},
            {"role": "user",   "content": user_content},
        ],
        temperature=settings.SYNTHESIS_TEMPERATURE,
        max_tokens=800,
    )

    full_text = resp.choices[0].message.content.strip()
    try:
        split_idx = full_text.rfind("\n{")
        if split_idx == -1:
            split_idx = full_text.rfind('{"confidence"')
        answer     = full_text[:split_idx].strip() if split_idx > 0 else full_text
        confidence = float(json.loads(full_text[split_idx:]).get("confidence", 0.0))
    except (json.JSONDecodeError, ValueError):
        answer, confidence = full_text, 0.5

    sources = [
        SourceDocument(
            title=d.source,
            excerpt=d.content[:200],
            url="",
            relevance=round(d.score, 3),
        )
        for d in all_docs[:3]
    ]
    return answer, round(min(max(confidence, 0.0), 1.0), 3), sources


# ── Main retrieval workflow ───────────────────────────────────────────────────

@workflow(name="retrieval_workflow")
async def retrieval_workflow(request: OrchestratorRequest) -> RetrievalResult:
    logger.info(
        "retrieval attempt=%d domain=%s tool=%s",
        request.attempt, request.domain, request.tool,
    )

    match request.tool:
        case "hyde":          docs = await run_hyde(request.query, request.domain)
        case "decomposition": docs = await run_decomposition(request.query, request.domain)
        case _:               docs = await run_hybrid(request.query, request.domain)

    # Parent-child: fetch parent chunks for context enrichment
    parent_ids  = list({d.parent_id for d in docs if d.parent_id})
    parent_docs = []
    for pid in parent_ids[:3]:
        parent = await asyncio.to_thread(fetch_parent_chunk, pid)
        if parent:
            parent_docs.append(parent)
    all_docs = docs + [p for p in parent_docs if p.id not in {d.id for d in docs}]

    # Pass conversation_id so synthesize_answer can inject short-term memory
    answer, confidence, source_docs = await synthesize_answer(
        request.query, all_docs, conversation_id=request.conversation_id
    )

    logger.info("retrieval complete confidence=%.3f", confidence)

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
        question_id=request.question_id,               # propagate
    )


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Retrieval Agent started.")
    yield
    logger.info("Retrieval Agent stopped.")


app = FastAPI(title="RAG Retrieval Agent — Local", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "agent": "retrieval"}


@app.post("/retrieve")
async def retrieve(raw: Request) -> Response:
    body    = await raw.json()
    request = OrchestratorRequest(**body)
    result_obj = await retrieval_workflow.run(request)
    outputs    = result_obj.get_outputs()
    result: RetrievalResult = outputs[0] if outputs else RetrievalResult(
        query=request.query,
        domain=request.domain,
        tool=request.tool,
        attempt=request.attempt,
        answer="Internal error.",
        confidence=0.0,
        sources=[],
        conversation_id=request.conversation_id,
        user_id=request.user_id,
        question_id=request.question_id,
    )
    return Response(
        content=json.dumps(result.__dict__),
        media_type="application/json",
    )


if __name__ == "__main__":
    uvicorn.run("agents.retrieval_agent:app", host="0.0.0.0", port=8002, reload=False)
