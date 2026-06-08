"""
Structured logging + Azure Application Insights via OpenTelemetry.

Key additions vs original:
  - CorrelationFilter   : injects request_id, agent, conversation_id, question_id
                          into every log record as structured fields — queryable
                          in Log Analytics via customDimensions.
  - configure_logging() : accepts agent_name so all log lines carry the agent label.
  - set_correlation()   : call at the start of each request to bind IDs to the
                          current thread/task context.

Log Analytics query examples:
  traces
  | where customDimensions.agent == "main"
  | where customDimensions.conversation_id == "abc-123"
  | order by timestamp asc

  exceptions
  | where customDimensions.question_id == "qst-3f1a2b4c"
"""
from __future__ import annotations

import logging
import sys
import threading
from contextvars import ContextVar

from shared.config import settings

# ── Per-request context vars (async-safe) ────────────────────────────────────
_request_id:     ContextVar[str] = ContextVar("request_id",     default="-")
_agent_name:     ContextVar[str] = ContextVar("agent_name",     default="-")
_conversation_id: ContextVar[str] = ContextVar("conversation_id", default="-")
_question_id:    ContextVar[str] = ContextVar("question_id",    default="-")


def set_correlation(
    *,
    request_id: str     = "-",
    agent: str          = "-",
    conversation_id: str = "-",
    question_id: str    = "-",
) -> None:
    """Bind correlation IDs to the current async context. Call at request entry."""
    _request_id.set(request_id)
    _agent_name.set(agent)
    _conversation_id.set(conversation_id)
    _question_id.set(question_id)


def get_correlation() -> dict:
    return {
        "request_id":      _request_id.get(),
        "agent":           _agent_name.get(),
        "conversation_id": _conversation_id.get(),
        "question_id":     _question_id.get(),
    }


# ── Log filter: injects correlation fields into every record ──────────────────

class CorrelationFilter(logging.Filter):
    """
    Adds request_id / agent / conversation_id / question_id to every LogRecord.
    These become customDimensions in Application Insights automatically via
    the OpenTelemetry exporter.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id      = _request_id.get()
        record.agent           = _agent_name.get()
        record.conversation_id = _conversation_id.get()
        record.question_id     = _question_id.get()
        return True


# ── Formatter: JSON with correlation fields ───────────────────────────────────

_JSON_FMT = (
    '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s",'
    '"agent":"%(agent)s","request_id":"%(request_id)s",'
    '"conversation_id":"%(conversation_id)s","question_id":"%(question_id)s",'
    '"msg":"%(message)s"}'
)


def configure_logging(agent_name: str = "unknown") -> None:
    """
    Call once at module startup.  agent_name labels all logs from this process
    (e.g. "main", "orchestrator", "retrieval").

    Azure Monitor / App Insights:
      Set APPLICATIONINSIGHTS_CONNECTION_STRING in .env (or ACA env vars).
      Logs flow to: Application Insights → Logs → 'traces' and 'exceptions' tables.
      customDimensions will include: agent, request_id, conversation_id, question_id.

    Log Analytics workspace queries (examples in module docstring).
    """
    # Bind agent name globally for this process
    _agent_name.set(agent_name)

    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_JSON_FMT))
    handler.addFilter(CorrelationFilter())

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers if configure_logging is called more than once
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)
    else:
        for h in root.handlers:
            h.addFilter(CorrelationFilter())
            h.setFormatter(logging.Formatter(_JSON_FMT))

    if settings.APPLICATIONINSIGHTS_CONNECTION_STRING:
        try:
            from azure.monitor.opentelemetry import configure_azure_monitor
            configure_azure_monitor(
                connection_string=settings.APPLICATIONINSIGHTS_CONNECTION_STRING,
                logger_name="",   # capture root logger → all agents
            )
            logging.getLogger(__name__).info(
                "Azure Monitor OpenTelemetry configured agent=%s", agent_name
            )
        except ImportError:
            logging.getLogger(__name__).warning(
                "azure-monitor-opentelemetry not installed — App Insights disabled."
            )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
