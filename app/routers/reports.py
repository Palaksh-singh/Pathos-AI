"""
Pathos AI — Reports Router
==============================
Generates a downloadable PDF summary of a chat session using reportlab.
The preview endpoint returns structured JSON (for the frontend's "neat
PDF preview" panel); the download endpoint returns the actual PDF bytes.
"""
from __future__ import annotations

import io
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.deps import get_current_user
from app.database import get_db_session
from app.models.db_models import ChatMessage, ChatSession, User
from app.schemas import ReportGenerateRequest, ReportPreview, ReportSection

logger = logging.getLogger("pathos_ai.routers.reports")
router = APIRouter(prefix="/reports", tags=["reports"])


async def _load_session_or_404(db: AsyncSession, user: User, session_id) -> ChatSession:
    result = await db.execute(
        select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user.id)
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    return session


def _build_sections(messages: list[ChatMessage], include_full_transcript: bool) -> list[ReportSection]:
    assistant_turns = [m for m in messages if m.role == "assistant"]
    summary_body = (
        "This report summarizes an educational conversation with Pathos AI. "
        f"The session included {len(assistant_turns)} assistant response(s). "
        "All content reflects general educational information only and does "
        "not constitute a diagnosis or treatment plan."
    )
    sections = [ReportSection(heading="Summary", body=summary_body)]

    if assistant_turns:
        sections.append(
            ReportSection(
                heading="Key Discussion Points",
                body="\n\n".join(f"• {m.content[:400]}" for m in assistant_turns[-5:]),
            )
        )

    if include_full_transcript:
        transcript = "\n\n".join(f"[{m.role.upper()}] {m.content}" for m in messages)
        sections.append(ReportSection(heading="Full Transcript (PII-masked)", body=transcript))

    return sections


@router.post("/preview", response_model=ReportPreview)
async def preview_report(
    payload: ReportGenerateRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> ReportPreview:
    session = await _load_session_or_404(db, user, payload.session_id)
    result = await db.execute(
        select(ChatMessage).where(ChatMessage.session_id == session.id).order_by(ChatMessage.created_at)
    )
    messages = list(result.scalars().all())

    return ReportPreview(
        title=f"Pathos AI Session Summary — {session.title}",
        generated_at=datetime.now(timezone.utc),
        patient_context_masked=None,
        sections=_build_sections(messages, payload.include_full_transcript),
        disclaimer=settings.disclaimer_text,
    )


@router.post("/download")
async def download_report(
    payload: ReportGenerateRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> StreamingResponse:
    preview = await preview_report(payload, user, db)
    pdf_bytes = _render_pdf(preview)

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="pathos-ai-report-{payload.session_id}.pdf"'},
    )


def _render_pdf(preview: ReportPreview) -> bytes:
    """
    Deferred reportlab import keeps this module (and the router file that
    imports it) importable in environments/tests that don't need PDF
    rendering, without adding reportlab as a hard import-time dependency.
    """
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=LETTER, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "PathosTitle", parent=styles["Title"], textColor=colors.HexColor("#0E7C86"), fontSize=20
    )
    heading_style = ParagraphStyle(
        "PathosHeading", parent=styles["Heading2"], textColor=colors.HexColor("#0F1B2D"), spaceBefore=14
    )
    body_style = ParagraphStyle("PathosBody", parent=styles["BodyText"], leading=15)
    disclaimer_style = ParagraphStyle(
        "PathosDisclaimer", parent=styles["BodyText"], textColor=colors.HexColor("#6B7280"), fontSize=8.5
    )

    elements = [
        Paragraph("Pathos AI", title_style),
        Paragraph(preview.title, styles["Heading3"]),
        Paragraph(f"Generated {preview.generated_at.strftime('%B %d, %Y at %H:%M UTC')}", styles["Normal"]),
        Spacer(1, 0.25 * inch),
    ]

    for section in preview.sections:
        elements.append(Paragraph(section.heading, heading_style))
        for paragraph in section.body.split("\n\n"):
            elements.append(Paragraph(paragraph.replace("\n", "<br/>"), body_style))
            elements.append(Spacer(1, 0.08 * inch))

    elements.append(Spacer(1, 0.3 * inch))
    elements.append(
        Table(
            [[Paragraph(preview.disclaimer, disclaimer_style)]],
            colWidths=[6.5 * inch],
            style=TableStyle(
                [
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7F9FB")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]
            ),
        )
    )

    doc.build(elements)
    return buffer.getvalue()
