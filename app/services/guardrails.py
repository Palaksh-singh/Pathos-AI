"""
Pathos AI — Clinical Guardrails
==================================
Two independent guardrail passes:

1. Input guardrail — screens the user's message BEFORE any retrieval or
   generation happens. Catches malicious prompt-injection attempts, requests
   for exact medication dosing, and crisis-risk language that needs an
   immediate resource response rather than a RAG answer.

2. Output guardrail — screens the LLM's *drafted* answer before it is
   streamed to the user. Catches definitive diagnostic claims ("you have X"),
   exact dosage instructions, and missing disclaimers. A failed output
   guardrail triggers a single bounded regeneration with a stricter system
   prompt (see `app/engines/llm_graph.py`); if the retry also fails, Pathos
   AI returns a safe, templated fallback rather than the raw model output.

Both passes are deliberately layered as (a) fast deterministic regex/keyword
rules that run on every request with near-zero latency, and (b) an optional
LLM-as-judge pass for the harder-to-pattern-match cases (wired in
`llm_graph.py` via `guardrail_judge_model`). This module implements (a) in
full and exposes the interface the judge pass plugs into.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.config import settings
from app.schemas import GuardrailFinding, GuardrailVerdict, RiskLevel

logger = logging.getLogger("pathos_ai.guardrails")


# ---------------------------------------------------------------------------
# Crisis / self-harm detection — highest priority, checked first
# ---------------------------------------------------------------------------

_CRISIS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(kill|hurt|harm)\s+myself\b", re.IGNORECASE),
    re.compile(r"\bsuicid(e|al)\b", re.IGNORECASE),
    re.compile(r"\bend(ing)?\s+my\s+life\b", re.IGNORECASE),
    re.compile(r"\bwant(ed)?\s+to\s+die\b", re.IGNORECASE),
    re.compile(r"\boverdos(e|ing)\s+on\s+purpose\b", re.IGNORECASE),
]

CRISIS_RESOURCE_MESSAGE = (
    "I'm concerned about what you've shared, and I want to make sure you get "
    "support from people who can help right now. If you're in the US, you can "
    "call or text 988 (Suicide & Crisis Lifeline) any time, or contact local "
    "emergency services if you're in immediate danger. If you're outside the "
    "US, please reach out to your local emergency number or a crisis line in "
    "your country. You don't have to go through this alone."
)


# ---------------------------------------------------------------------------
# Prompt-injection / malicious-intent detection (input guardrail)
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(?:all\s+|any\s+|previous\s+|the\s+)*(?:instructions|prompt)", re.IGNORECASE),
    re.compile(r"you are now (a|an|in) (jailbreak|dan|unfiltered)", re.IGNORECASE),
    re.compile(r"disregard (your|the) (system prompt|guardrails|safety)", re.IGNORECASE),
    re.compile(r"pretend (you|to be) (a|an)? ?(real )?(doctor|physician)", re.IGNORECASE),
    re.compile(r"act as (a|an) unrestricted", re.IGNORECASE),
]

_EXACT_DOSAGE_REQUEST_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"how many (mg|milligrams|pills|tablets)\s+.*(should|do)\s+i\s+(take|use)", re.IGNORECASE),
    re.compile(r"exact dos(age|e)\s+of", re.IGNORECASE),
    re.compile(r"lethal dos(age|e)", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Output guardrail: definitive diagnosis / exact dosage in the LLM's draft
# ---------------------------------------------------------------------------

_DEFINITIVE_DIAGNOSIS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\byou (definitely |certainly )?have\s+[A-Za-z][A-Za-z\s]{2,40}\b", re.IGNORECASE),
    re.compile(r"\byou are (definitely |certainly )?diagnosed with\b", re.IGNORECASE),
    re.compile(r"\bthis (is|confirms)\s+(definitely\s+)?[A-Za-z][A-Za-z\s]{2,40}\b(disease|disorder|syndrome|cancer)\b", re.IGNORECASE),
    re.compile(r"\bi (can\s+)?confirm you have\b", re.IGNORECASE),
]

_EXACT_DOSAGE_INSTRUCTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\btake\s+\d+(\.\d+)?\s*(mg|milligrams|ml|tablets|pills)\b", re.IGNORECASE),
    re.compile(r"\byou should take\s+\d+(\.\d+)?\s*(mg|milligrams|ml)\b", re.IGNORECASE),
]

_REQUIRED_DISCLAIMER_MARKERS = ("educational", "not a substitute", "licensed clinician", "professional medical")


@dataclass
class GuardrailResult:
    verdict: GuardrailVerdict
    risk_level: RiskLevel
    findings: list[GuardrailFinding]
    safe_response_override: str | None = None


class InputGuardrail:
    """Screens raw (already PII-masked) user input before retrieval/generation."""

    def evaluate(self, masked_message: str) -> GuardrailResult:
        findings: list[GuardrailFinding] = []

        # 1. Crisis detection takes absolute priority
        if settings.enable_crisis_detection:
            for pattern in _CRISIS_PATTERNS:
                match = pattern.search(masked_message)
                if match:
                    findings.append(
                        GuardrailFinding(
                            rule_name="crisis_language_detected",
                            verdict=GuardrailVerdict.BLOCKED,
                            reason="Message contains language indicating potential self-harm risk.",
                            matched_span=match.group(0),
                        )
                    )
                    logger.warning("crisis_language_detected", extra={"rule": "crisis"})
                    return GuardrailResult(
                        verdict=GuardrailVerdict.BLOCKED,
                        risk_level=RiskLevel.CRISIS,
                        findings=findings,
                        safe_response_override=CRISIS_RESOURCE_MESSAGE,
                    )

        # 2. Prompt injection / jailbreak attempts
        for pattern in _INJECTION_PATTERNS:
            match = pattern.search(masked_message)
            if match:
                findings.append(
                    GuardrailFinding(
                        rule_name="prompt_injection_detected",
                        verdict=GuardrailVerdict.BLOCKED,
                        reason="Message attempts to override system instructions or safety configuration.",
                        matched_span=match.group(0),
                    )
                )
                return GuardrailResult(
                    verdict=GuardrailVerdict.BLOCKED,
                    risk_level=RiskLevel.MODERATE,
                    findings=findings,
                    safe_response_override=(
                        "I can't override my safety configuration or role. I'm happy to help "
                        "with general, educational health information within those bounds — "
                        "what would you like to know?"
                    ),
                )

        # 3. Exact dosage requests — not blocked outright, but flagged so the
        #    generation node can inject an explicit "no exact dosing" instruction.
        for pattern in _EXACT_DOSAGE_REQUEST_PATTERNS:
            match = pattern.search(masked_message)
            if match:
                findings.append(
                    GuardrailFinding(
                        rule_name="exact_dosage_request",
                        verdict=GuardrailVerdict.ALLOWED,
                        reason="User is requesting specific dosage guidance; generation must stay general.",
                        matched_span=match.group(0),
                    )
                )
                return GuardrailResult(
                    verdict=GuardrailVerdict.ALLOWED,
                    risk_level=RiskLevel.LOW,
                    findings=findings,
                )

        return GuardrailResult(verdict=GuardrailVerdict.ALLOWED, risk_level=RiskLevel.NONE, findings=findings)


class OutputGuardrail:
    """Screens the LLM's drafted answer before it is streamed to the user."""

    def evaluate(self, draft_answer: str) -> GuardrailResult:
        findings: list[GuardrailFinding] = []

        for pattern in _DEFINITIVE_DIAGNOSIS_PATTERNS:
            match = pattern.search(draft_answer)
            if match:
                findings.append(
                    GuardrailFinding(
                        rule_name="definitive_diagnosis_claim",
                        verdict=GuardrailVerdict.REGENERATE,
                        reason="Draft answer makes a definitive diagnostic claim.",
                        matched_span=match.group(0),
                    )
                )
                return GuardrailResult(
                    verdict=GuardrailVerdict.REGENERATE,
                    risk_level=RiskLevel.MODERATE,
                    findings=findings,
                )

        for pattern in _EXACT_DOSAGE_INSTRUCTION_PATTERNS:
            match = pattern.search(draft_answer)
            if match:
                findings.append(
                    GuardrailFinding(
                        rule_name="exact_dosage_instruction",
                        verdict=GuardrailVerdict.REGENERATE,
                        reason="Draft answer specifies an exact medication dosage.",
                        matched_span=match.group(0),
                    )
                )
                return GuardrailResult(
                    verdict=GuardrailVerdict.REGENERATE,
                    risk_level=RiskLevel.MODERATE,
                    findings=findings,
                )

        if not any(marker in draft_answer.lower() for marker in _REQUIRED_DISCLAIMER_MARKERS):
            findings.append(
                GuardrailFinding(
                    rule_name="missing_disclaimer",
                    verdict=GuardrailVerdict.ALLOWED,
                    reason="Draft answer is missing the clinical disclaimer; will be auto-appended.",
                )
            )
            # Not a regenerate — cheaper to append deterministically than to
            # burn another generation call. Handled in llm_graph.py.

        return GuardrailResult(verdict=GuardrailVerdict.ALLOWED, risk_level=RiskLevel.NONE, findings=findings)


input_guardrail = InputGuardrail()
output_guardrail = OutputGuardrail()
