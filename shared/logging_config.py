"""
Structured logging + optional Azure Application Insights via OpenTelemetry.
"""
from __future__ import annotations

import logging
import sys

from shared.config import settings


def configure_logging() -> None:
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
    )
    # To see retrieval diagnostics (chunks pulled, scores, raw LLM output):
    #   set LOG_LEVEL=DEBUG in .env
    # WARNING: DEBUG logs are verbose — use only when diagnosing retrieval failures.

    if settings.APPLICATIONINSIGHTS_CONNECTION_STRING:
        try:
            from azure.monitor.opentelemetry import configure_azure_monitor
            configure_azure_monitor(
                connection_string=settings.APPLICATIONINSIGHTS_CONNECTION_STRING
            )
            logging.getLogger(__name__).info("Azure Monitor OpenTelemetry configured.")
        except ImportError:
            logging.getLogger(__name__).warning(
                "azure-monitor-opentelemetry not installed — skipping App Insights."
            )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
