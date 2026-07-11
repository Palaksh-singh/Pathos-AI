"""
Pathos AI — FastAPI Application Entrypoint
==============================================
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.core.logging_config import configure_logging
from app.core.telemetry import configure_langsmith
from app.database import init_db
from app.routers import auth, chat, reports

configure_logging()
logger = logging.getLogger("pathos_ai.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("pathos_ai_starting", extra={"environment": settings.environment.value})
    configure_langsmith()
    await init_db()
    yield
    logger.info("pathos_ai_shutting_down")


app = FastAPI(
    title=settings.app_name,
    description=(
        "Pathos AI — a stateful, guardrail-enforced medical RAG assistant. "
        "Educational demo system: not a certified medical device and not a "
        "substitute for professional clinical judgment."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs" if not settings.is_production else None,
    redoc_url="/api/redoc" if not settings.is_production else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request-scoped correlation ID + timing middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    start = time.perf_counter()

    response = await call_next(request)

    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Response-Time-Ms"] = f"{duration_ms:.1f}"

    logger.info(
        "http_request_completed",
        extra={
            "correlation_id": correlation_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 1),
        },
    )
    return response


# ---------------------------------------------------------------------------
# Exception handlers — never leak stack traces or raw internals to clients
# ---------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.warning("request_validation_failed", extra={"path": request.url.path, "errors": exc.errors()})
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "One or more fields failed validation.", "errors": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_exception", extra={"path": request.url.path})
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. Pathos AI has logged this issue."},
    )


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(auth.router, prefix=settings.api_v1_prefix)
app.include_router(chat.router, prefix=settings.api_v1_prefix)
app.include_router(reports.router, prefix=settings.api_v1_prefix)


@app.get("/api/health", tags=["system"])
async def health_check() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name, "environment": settings.environment.value}


# Serve the static frontend (Tailwind + vanilla JS SPA) if present.
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
