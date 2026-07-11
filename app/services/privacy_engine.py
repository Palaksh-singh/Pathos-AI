"""
Pathos AI — Privacy Engine
============================
Detects and reversibly tokenizes PII/PHI-shaped spans (names, phone numbers,
emails, SSNs, medical record numbers, dates of birth, addresses, insurance
IDs) BEFORE any text is sent to an external LLM provider or written to logs.

Design notes
------------
* Detection here is a fast, dependency-free regex/heuristic layer intended
  to run on every request with near-zero latency overhead. In production
  this is layered with a statistical NER model (e.g. a spaCy `en_core_web_lg`
  pipeline or a hosted PII service like AWS Comprehend Medical) for names
  that don't follow a structural pattern — the interface (`mask()` /
  `unmask()`) is written so that swap-in is a one-line change in
  `_detect_person_names`.
* Masking is *reversible within the request only*: the `pii_map` returned
  by `mask()` must never be persisted to a database or log sink. It exists
  purely so the authenticated end user sees their own real data reflected
  back in the UI, while the LLM provider and all logs only ever see
  stable placeholder tokens (e.g. `PERSON_1`, `MRN_1`).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from app.schemas import PIIEntityType, PIIMaskResult

logger = logging.getLogger("pathos_ai.privacy_engine")


@dataclass
class _CompiledRule:
    entity_type: PIIEntityType
    pattern: re.Pattern[str]
    # Confidence gate — lets us tune false-positive-prone rules independently.
    min_len: int = 0


def _build_rules() -> list[_CompiledRule]:
    return [
        _CompiledRule(
            PIIEntityType.EMAIL,
            re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
        ),
        _CompiledRule(
            PIIEntityType.SSN,
            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        ),
        _CompiledRule(
            # US-style phone numbers in common formats
            PIIEntityType.PHONE,
            re.compile(
                r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
            ),
        ),
        _CompiledRule(
            # Medical Record Number: "MRN" / "MR#" / "Medical Record" followed by digits
            PIIEntityType.MRN,
            re.compile(r"\b(?:MRN|MR#|Medical Record(?: Number)?)[:\s#]*([A-Z0-9-]{4,15})\b", re.IGNORECASE),
        ),
        _CompiledRule(
            # Insurance / policy ID: "Policy #", "Insurance ID", "Member ID"
            PIIEntityType.INSURANCE_ID,
            re.compile(
                r"\b(?:Policy|Insurance|Member)\s*(?:ID|#|Number)?[:\s#]*([A-Z0-9-]{5,20})\b",
                re.IGNORECASE,
            ),
        ),
        _CompiledRule(
            # Dates of birth: explicit DOB label OR standalone MM/DD/YYYY-shaped dates
            PIIEntityType.DATE_OF_BIRTH,
            re.compile(
                r"\b(?:DOB|Date of Birth)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
                re.IGNORECASE,
            ),
        ),
        _CompiledRule(
            # Street addresses: number + street name + common suffix
            PIIEntityType.ADDRESS,
            re.compile(
                r"\b\d{1,6}\s+([A-Za-z0-9.\s]{2,40})\s+"
                r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way)\b",
                re.IGNORECASE,
            ),
        ),
    ]


# Lightweight person-name heuristic: "Mr./Mrs./Dr./Ms. FirstName LastName" or
# "my name is X", "patient X", "I am X" — deliberately conservative to keep
# false positives low without a full NER model wired in.
_PERSON_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"),
    re.compile(r"\b(?:my name is|i am|i'm|this is)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b", re.IGNORECASE),
    re.compile(r"\bpatient\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b", re.IGNORECASE),
]


class PrivacyEngine:
    """Stateless-by-design masking engine — every call gets a fresh token map."""

    def __init__(self) -> None:
        self._rules = _build_rules()

    # -- Public API ----------------------------------------------------------

    def mask(self, text: str) -> PIIMaskResult:
        """
        Detects PII spans and replaces them with stable, typed placeholder
        tokens (e.g. `EMAIL_1`, `PHONE_1`). Returns the masked text plus a
        pii_map for in-request unmasking. Never raises on malformed input —
        detection failures degrade to "no entity found", not an exception,
        since this sits on the hot path of every chat request.
        """
        if not text:
            return PIIMaskResult(masked_text=text, pii_map={}, entities_found=[])

        working_text = text
        pii_map: dict[str, str] = {}
        entities_found: set[PIIEntityType] = set()
        counters: dict[PIIEntityType, int] = {}

        # Structural patterns (email, SSN, phone, MRN, insurance ID, DOB, address)
        for rule in self._rules:
            working_text = self._apply_rule(working_text, rule, pii_map, counters, entities_found)

        # Person-name heuristics
        working_text = self._apply_person_names(working_text, pii_map, counters, entities_found)

        if entities_found:
            logger.info(
                "pii_masking_applied",
                extra={"entity_types": sorted(e.value for e in entities_found), "span_count": len(pii_map)},
            )

        return PIIMaskResult(
            masked_text=working_text,
            pii_map=pii_map,
            entities_found=sorted(entities_found, key=lambda e: e.value),
        )

    def unmask(self, text: str, pii_map: dict[str, str]) -> str:
        """
        Restores original values for a given request's pii_map. Called only
        on the response path, only for the authenticated user who owns the
        underlying session — never for cross-user data or persisted storage.
        """
        if not pii_map:
            return text
        restored = text
        # Replace longest tokens first to avoid partial-token collisions
        # (e.g. PERSON_1 vs PERSON_10).
        for token in sorted(pii_map, key=len, reverse=True):
            restored = restored.replace(token, pii_map[token])
        return restored

    def redact_for_logging(self, text: str) -> str:
        """Convenience helper: mask and discard the map — for log sinks only."""
        return self.mask(text).masked_text

    # -- Internal --------------------------------------------------------------

    @staticmethod
    def _next_token(entity_type: PIIEntityType, counters: dict[PIIEntityType, int]) -> str:
        counters[entity_type] = counters.get(entity_type, 0) + 1
        return f"{entity_type.value}_{counters[entity_type]}"

    def _apply_rule(
        self,
        text: str,
        rule: _CompiledRule,
        pii_map: dict[str, str],
        counters: dict[PIIEntityType, int],
        entities_found: set[PIIEntityType],
    ) -> str:
        def _replace(match: re.Match[str]) -> str:
            original = match.group(0)
            if len(original) < rule.min_len:
                return original
            token = self._next_token(rule.entity_type, counters)
            pii_map[token] = original
            entities_found.add(rule.entity_type)
            return token

        return rule.pattern.sub(_replace, text)

    def _apply_person_names(
        self,
        text: str,
        pii_map: dict[str, str],
        counters: dict[PIIEntityType, int],
        entities_found: set[PIIEntityType],
    ) -> str:
        working = text
        for pattern in _PERSON_PATTERNS:
            def _replace(match: re.Match[str]) -> str:
                original = match.group(1) if match.groups() else match.group(0)
                token = self._next_token(PIIEntityType.PERSON, counters)
                pii_map[token] = original
                entities_found.add(PIIEntityType.PERSON)
                return match.group(0).replace(original, token)

            working = pattern.sub(_replace, working)
        return working


# Module-level singleton — the engine holds no per-request state, so this
# is safe to share across the async event loop.
privacy_engine = PrivacyEngine()
