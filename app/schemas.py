"""
Pathos AI — Pydantic v2 Schemas
=================================
Strict request/response contracts. These models are the single source of
truth for what crosses the API boundary; nothing reaches a service function
without passing through one of these first.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=10, max_length=128)
    full_name: str = Field(min_length=1, max_length=120)

    @field_validator("password")
    @classmethod
    def _password_strength(cls, v: str) -> str:
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter.")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter.")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one digit.")
        if not re.search(r"[^\w\s]", v):
            raise ValueError("Password must contain at least one special character.")
        return v


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    email: EmailStr
    full_name: str
    created_at: datetime


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in_seconds: int


class TokenPayload(BaseModel):
    sub: str  # user id
    exp: int
    iat: int
    type: str  # "access" | "refresh"
    jti: str


# ---------------------------------------------------------------------------
# Chat / clinical domain
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    CRISIS = "crisis"


class GuardrailVerdict(str, Enum):
    ALLOWED = "allowed"
    BLOCKED = "blocked"
    REGENERATE = "regenerate"


class ChatMessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatRequest(BaseModel):
    session_id: uuid.UUID | None = None
    message: str = Field(min_length=1, max_length=4000)

    @field_validator("message")
    @classmethod
    def _strip_and_bound(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Message cannot be empty or whitespace only.")
        return cleaned


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_title: str
    source_url: str | None = None
    text: str
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: float = 0.0


class GuardrailFinding(BaseModel):
    rule_name: str
    verdict: GuardrailVerdict
    reason: str
    matched_span: str | None = None


class TraceStep(BaseModel):
    """One node execution in the LangGraph pipeline — surfaced to the UI trace panel."""
    node_name: str
    started_at: datetime
    duration_ms: float
    status: str  # "ok" | "blocked" | "retried" | "error"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    session_id: uuid.UUID
    message_id: uuid.UUID
    answer: str
    citations: list[RetrievedChunk] = Field(default_factory=list)
    disclaimer: str
    risk_level: RiskLevel = RiskLevel.NONE
    guardrail_findings: list[GuardrailFinding] = Field(default_factory=list)
    trace: list[TraceStep] = Field(default_factory=list)
    tokens_input: int = 0
    tokens_output: int = 0
    latency_ms: float = 0.0


class ChatSessionSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

class ReportGenerateRequest(BaseModel):
    session_id: uuid.UUID
    include_full_transcript: bool = False


class ReportSection(BaseModel):
    heading: str
    body: str


class ReportPreview(BaseModel):
    title: str
    generated_at: datetime
    patient_context_masked: str | None = None
    sections: list[ReportSection]
    disclaimer: str


# ---------------------------------------------------------------------------
# Privacy engine internal contracts
# ---------------------------------------------------------------------------

class PIIEntityType(str, Enum):
    PERSON = "PERSON"
    PHONE = "PHONE"
    EMAIL = "EMAIL"
    SSN = "SSN"
    MRN = "MRN"
    DATE_OF_BIRTH = "DOB"
    ADDRESS = "ADDRESS"
    INSURANCE_ID = "INSURANCE_ID"


class PIIMaskResult(BaseModel):
    masked_text: str
    pii_map: dict[str, str]  # token -> original value (kept in-process only)
    entities_found: list[PIIEntityType] = Field(default_factory=list)
