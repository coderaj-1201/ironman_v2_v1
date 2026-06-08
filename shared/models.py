"""
Shared typed models for the RAG pipeline.

Changes vs original:
  - UserQuery        : + question_id (auto-generated)
  - OrchestratorRequest : + question_id (propagated)
  - RetrievalResult  : + question_id (propagated)
  - FinalResponse    : answer_id already existed; + question_id added
  - FeedbackRequest  : NEW — payload for POST /feedback
  - MemoryRecord     : NEW — long-term memory entry
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from uuid import uuid4


class Domain(StrEnum):
    HR    = "hr"
    LEGAL = "legal"
    IT    = "it"


class RetrievalTool(StrEnum):
    HYBRID        = "hybrid"
    HYDE          = "hyde"
    DECOMPOSITION = "decomposition"


@dataclass
class UserQuery:
    text: str
    conversation_id: str
    user_id: str
    question_id: str = field(default_factory=lambda: f"q-{uuid4().hex[:8]}")


@dataclass
class OrchestratorRequest:
    query: str
    domain: Domain
    tool: RetrievalTool
    attempt: int
    conversation_id: str
    user_id: str
    question_id: str = ""


@dataclass
class SourceDocument:
    title: str
    excerpt: str
    url: str     = ""
    relevance: float = 0.0


@dataclass
class RetrievalResult:
    query: str
    domain: Domain
    tool: RetrievalTool
    attempt: int
    answer: str
    confidence: float
    sources: list[dict]          # serialised SourceDocument dicts
    conversation_id: str
    user_id: str
    question_id: str = ""

    @property
    def passed(self) -> bool:
        from shared.config import settings
        return self.confidence >= settings.CONFIDENCE_THRESHOLD


@dataclass
class FinalResponse:
    status: str                  # "success" | "failure"
    answer: str
    domain: Domain | None
    sources: list[dict]  = field(default_factory=list)
    confidence: float    = 0.0
    attempts_used: int   = 0
    conversation_id: str = ""
    user_id: str         = ""
    question_id: str     = ""
    answer_id: str       = field(default_factory=lambda: f"ans-{uuid4().hex[:8]}")


# ── New models ────────────────────────────────────────────────────────────────

@dataclass
class FeedbackRequest:
    answer_id: str
    question_id: str
    user_id: str
    rating: int           # 1–5
    comment: str = ""
    feedback_id: str = field(default_factory=lambda: f"fb-{uuid4().hex[:8]}")


@dataclass
class MemoryRecord:
    user_id: str
    content: str                 # extracted fact / preference
    memory_type: str = "long"    # "long" only (short-term is derived from conversations)
    memory_id: str   = field(default_factory=lambda: f"mem-{uuid4().hex[:8]}")
