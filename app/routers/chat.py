"""
Pathos AI — Chat Router
===========================
Exposes the streaming chat endpoint. The LangGraph workflow itself runs to
completion internally (guardrails need the *full* draft before it's safe to
release any tokens — you cannot un-stream a definitive-diagnosis sentence
once it's on the user's screen), then the final, validated answer is
streamed to the client word-by-word over Server-Sent Events so the UI still
gets the "live typing" feel the product brief calls for.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.database import get_db_session
from app.engines.llm_graph import run_pathos_graph
from app.models.db_models import ChatMessage, ChatSession, GuardrailAuditLog, User
from app.schemas import ChatRequest, ChatResponse, ChatSessionSummary, GuardrailFinding

logger = logging.getLogger("pathos_ai.routers.chat")
router = APIRouter(prefix="/chat", tags=["chat"])


async def _get_or_create_session(
    db: AsyncSession, user: User, session_id: uuid.UUID | None
) -> ChatSession:
    if session_id is not None:
        result = await db.execute(
            select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user.id)
        )
        session = result.scalar_one_or_none()
        if session is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found.")
        return session

    session = ChatSession(user_id=user.id, title="New conversation")
    db.add(session)
    await db.flush()
    return session


async def _load_history(db: AsyncSession, session_id: uuid.UUID, limit: int = 12) -> list[dict[str, str]]:
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(limit)
    )
    messages = list(reversed(result.scalars().all()))
    return [{"role": m.role, "content": m.content} for m in messages]


@router.post("/stream")
async def chat_stream(
    payload: ChatRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> StreamingResponse:
    session = await _get_or_create_session(db, user, payload.session_id)
    history = await _load_history(db, session.id)

    start = time.perf_counter()
    try:
        result_state = await run_pathos_graph(session.id, payload.message, history)
    except Exception:
        logger.exception("chat_graph_execution_failed", extra={"session_id": str(session.id)})
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Pathos AI could not process this message. Please try again.",
        )
    latency_ms = (time.perf_counter() - start) * 1000

    final_answer = result_state.get("final_answer", "")

    # Persist masked user turn + assistant turn + guardrail audit trail.
    # Content stored is always the masked message — see privacy_engine.py.
    user_message = ChatMessage(
        session_id=session.id,
        role="user",
        content=result_state.get("masked_message", payload.message),
        risk_level=result_state.get("risk_level", "none").value
        if hasattr(result_state.get("risk_level"), "value")
        else str(result_state.get("risk_level", "none")),
    )
    assistant_message = ChatMessage(
        session_id=session.id,
        role="assistant",
        content=final_answer,
        latency_ms=int(latency_ms),
    )
    db.add_all([user_message, assistant_message])

    for finding in result_state.get("guardrail_findings", []):
        f: GuardrailFinding = finding
        db.add(
            GuardrailAuditLog(
                session_id=session.id,
                rule_name=f.rule_name,
                verdict=f.verdict.value if hasattr(f.verdict, "value") else str(f.verdict),
                reason=f.reason,
            )
        )

    if session.title == "New conversation":
        session.title = payload.message[:60] + ("…" if len(payload.message) > 60 else "")

    await db.commit()

    async def event_stream():
        # Header event: session + trace metadata the UI needs immediately.
        header = {
            "type": "meta",
            "session_id": str(session.id),
            "message_id": str(assistant_message.id),
            "risk_level": str(result_state.get("risk_level", "none")),
            "trace": [step.model_dump(mode="json") for step in result_state.get("trace", [])],
            "citations": [c.model_dump(mode="json") for c in result_state.get("reranked_chunks", [])],
            "latency_ms": round(latency_ms, 1),
        }
        yield f"data: {json.dumps(header)}\n\n"

        # Stream the validated final answer token-by-word for a live-typing feel.
        words = final_answer.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
            await asyncio.sleep(0.012)

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/sessions", response_model=list[ChatSessionSummary])
async def list_sessions(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> list[ChatSession]:
    result = await db.execute(
        select(ChatSession).where(ChatSession.user_id == user.id).order_by(ChatSession.updated_at.desc())
    )
    sessions = result.scalars().all()
    summaries = []
    for s in sessions:
        count_result = await db.execute(select(ChatMessage).where(ChatMessage.session_id == s.id))
        summaries.append(
            ChatSessionSummary(
                id=s.id,
                title=s.title,
                created_at=s.created_at,
                updated_at=s.updated_at,
                message_count=len(count_result.scalars().all()),
            )
        )
    return summaries
