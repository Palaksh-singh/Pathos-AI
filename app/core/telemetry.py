"""
Pathos AI — Observability & Tracing
======================================
Wraps every LangGraph node with:
  1. A lightweight in-process trace step (appended to `state["trace"]`) that
     the frontend renders as the live "Pipeline Trace" panel.
  2. An OpenTelemetry span, exported to whatever OTLP collector is
     configured (Arize Phoenix, Honeycomb, Jaeger, etc.) via
     OTEL_EXPORTER_ENDPOINT.
  3. LangSmith run tracing, enabled purely via environment variables
     (LANGSMITH_TRACING / LANGSMITH_API_KEY / LANGSMITH_PROJECT) — LangChain
     and LangGraph pick these up natively, so `configure_langsmith()` below
     just validates and exports them predictably at startup.
"""
from __future__ import annotations

import functools
import logging
import os
import time
from datetime import datetime, timezone
from typing import Awaitable, Callable, TypeVar

from app.config import settings

logger = logging.getLogger("pathos_ai.telemetry")

_T = TypeVar("_T")

_tracer = None


def _get_tracer():
    """Lazily builds an OpenTelemetry tracer; falls back to a no-op if the
    otel SDK isn't installed / configured, so tracing is always optional."""
    global _tracer
    if _tracer is not None:
        return _tracer

    if not settings.otel_exporter_endpoint:
        _tracer = _NoOpTracer()
        return _tracer

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource(attributes={SERVICE_NAME: settings.otel_service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        _tracer = trace.get_tracer(settings.otel_service_name)
        logger.info("otel_tracer_initialized", extra={"endpoint": settings.otel_exporter_endpoint})
    except Exception:
        logger.exception("otel_tracer_init_failed_falling_back_to_noop")
        _tracer = _NoOpTracer()

    return _tracer


class _NoOpSpan:
    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def set_attribute(self, *args: object, **kwargs: object) -> None:
        return None


class _NoOpTracer:
    def start_as_current_span(self, *_args: object, **_kwargs: object) -> _NoOpSpan:
        return _NoOpSpan()


def configure_langsmith() -> None:
    """Exports LangSmith env vars in the exact names LangChain/LangGraph expect."""
    if not settings.langsmith_tracing:
        return
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key.get_secret_value()
    logger.info("langsmith_tracing_enabled", extra={"project": settings.langsmith_project})


def traced_node(node_name: str) -> Callable[[Callable[..., Awaitable[_T]]], Callable[..., Awaitable[_T]]]:
    """
    Decorator applied to every LangGraph node function. Records:
      - wall-clock duration
      - node status (ok / blocked / retried / error)
      - an OpenTelemetry span (if configured)
    and appends a `TraceStep` to the graph state's `trace` list so the
    frontend can render step-by-step pipeline execution in real time.
    """

    def decorator(fn: Callable[..., Awaitable[_T]]) -> Callable[..., Awaitable[_T]]:
        @functools.wraps(fn)
        async def wrapper(state, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            from app.schemas import TraceStep  # local import avoids circular import at module load

            tracer = _get_tracer()
            started_at = datetime.now(timezone.utc)
            start_perf = time.perf_counter()
            status = "ok"

            try:
                with tracer.start_as_current_span(f"pathos_ai.node.{node_name}") as span:
                    span.set_attribute("pathos.node_name", node_name)
                    result_state = await fn(state, *args, **kwargs)

                    if isinstance(result_state, dict) and result_state.get("blocked"):
                        status = "blocked"
                    if isinstance(result_state, dict) and result_state.get("regeneration_count", 0) > (
                        state.get("regeneration_count", 0) if isinstance(state, dict) else 0
                    ):
                        status = "retried"

                    return result_state
            except Exception:
                status = "error"
                logger.exception("graph_node_failed", extra={"node": node_name})
                raise
            finally:
                duration_ms = (time.perf_counter() - start_perf) * 1000
                try:
                    step = TraceStep(
                        node_name=node_name,
                        started_at=started_at,
                        duration_ms=round(duration_ms, 2),
                        status=status,
                    )
                    if isinstance(state, dict):
                        state.setdefault("trace", [])
                        state["trace"].append(step)
                except Exception:
                    logger.exception("trace_step_append_failed", extra={"node": node_name})

        return wrapper

    return decorator
