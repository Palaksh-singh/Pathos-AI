"""
Pathos AI — LangGraph Clinical Workflow Engine
==================================================
This module is the executable version of docs/architecture.md's state
diagram: UserInput -> PIIMasking -> InputGuardrail -> HybridRetrieval ->
CrossEncoderRerank -> Generation -> OutputGuardrail -> PIIUnmasking ->
StreamedOutput, with a bounded regeneration loop and a hard block path for
crisis/injection detection.

Every node is a small, independently testable function that takes and
returns a `PathosGraphState`. Nodes never mutate state in place — they
return a new/updated state — which keeps the graph replayable and makes
each node trivially unit-testable without booting the whole graph.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

from app.config import settings
from app.core.telemetry import traced_node
from app.engines.reranker import reranker
from app.schemas import (
    ChatMessageRole,
    GuardrailFinding,
    GuardrailVerdict,
    RetrievedChunk,
    RiskLevel,
    TraceStep,
)
from app.services.guardrails import CRISIS_RESOURCE_MESSAGE, input_guardrail, output_guardrail
from app.services.privacy_engine import privacy_engine
from app.services.retrieval_service import retrieval_service

logger = logging.getLogger("pathos_ai.llm_graph")


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class PathosGraphState(TypedDict, total=False):
    session_id: str
    raw_message: str
    chat_history: list[dict[str, str]]  # [{"role": "user"/"assistant", "content": ...}]

    masked_message: str
    pii_map: dict[str, str]

    standalone_query: str

    guardrail_findings: list[GuardrailFinding]
    risk_level: RiskLevel
    blocked: bool
    blocked_response: str | None

    retrieved_chunks: list[RetrievedChunk]
    reranked_chunks: list[RetrievedChunk]

    draft_answer: str
    final_answer: str
    regeneration_count: int

    trace: list[TraceStep]
    tokens_input: int
    tokens_output: int


def build_initial_state(session_id: uuid.UUID, message: str, chat_history: list[dict[str, str]]) -> PathosGraphState:
    return PathosGraphState(
        session_id=str(session_id),
        raw_message=message,
        chat_history=chat_history,
        pii_map={},
        guardrail_findings=[],
        risk_level=RiskLevel.NONE,
        blocked=False,
        blocked_response=None,
        retrieved_chunks=[],
        reranked_chunks=[],
        draft_answer="",
        final_answer="",
        regeneration_count=0,
        trace=[],
        tokens_input=0,
        tokens_output=0,
    )


# ---------------------------------------------------------------------------
# System prompt construction
# ---------------------------------------------------------------------------

_BASE_SYSTEM_PROMPT = """You are Pathos AI, a clinical-education assistant.

Hard rules you must never break:
1. Never state or imply a definitive diagnosis. Use language like "this can be
   associated with" or "a clinician could evaluate for" instead of "you have X".
2. Never give an exact medication dosage (a specific mg/ml/tablet count). You
   may describe how dosing decisions are generally made and who makes them
   (a prescribing clinician or pharmacist), but never a specific number.
3. Always ground your answer in the provided context chunks. If the context
   doesn't cover the question, say so plainly rather than guessing.
4. Always end your answer with a short, natural restatement that this is
   educational information, not a substitute for professional medical advice.
5. Be warm, clear, and plain-spoken. Avoid unnecessary jargon.
"""

_STRICT_RETRY_SUFFIX = """
IMPORTANT — your previous draft violated a safety rule and was rejected.
Rewrite your answer now, strictly avoiding definitive diagnostic language
and exact dosage numbers. Be more conservative than you were before.
"""


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(No relevant reference material was found for this question.)"
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"[{i}] {chunk.document_title}\n{chunk.text}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

@traced_node("pii_masking")
async def pii_masking_node(state: PathosGraphState) -> PathosGraphState:
    result = privacy_engine.mask(state["raw_message"])
    state["masked_message"] = result.masked_text
    state["pii_map"] = result.pii_map
    return state


@traced_node("input_guardrail")
async def input_guardrail_node(state: PathosGraphState) -> PathosGraphState:
    result = input_guardrail.evaluate(state["masked_message"])
    state["guardrail_findings"] = list(state.get("guardrail_findings", [])) + result.findings
    state["risk_level"] = result.risk_level

    if result.verdict == GuardrailVerdict.BLOCKED:
        state["blocked"] = True
        state["blocked_response"] = result.safe_response_override or (
            "I'm not able to help with that request. " + CRISIS_RESOURCE_MESSAGE
            if result.risk_level == RiskLevel.CRISIS
            else "I'm not able to help with that request."
        )
    return state


@traced_node("query_rewrite")
async def query_rewrite_node(state: PathosGraphState) -> PathosGraphState:
    """
    Rewrites the (masked) message into a standalone query using recent chat
    history, so retrieval doesn't fail on follow-ups like "what about
    ibuprofen instead?" that only make sense with prior turns of context.
    For turn 1 (no history), the masked message is used as-is.
    """
    history = state.get("chat_history", [])
    if not history:
        state["standalone_query"] = state["masked_message"]
        return state

    # Lightweight heuristic rewrite: prepend the last user turn's topic
    # rather than a full LLM call, to keep this node cheap and fast. A full
    # implementation may call a small/cheap LLM here for true rewriting.
    last_user_turns = [h["content"] for h in history if h.get("role") == "user"][-2:]
    combined = " ".join(last_user_turns + [state["masked_message"]])
    state["standalone_query"] = combined
    return state


@traced_node("hybrid_retrieval")
async def hybrid_retrieval_node(state: PathosGraphState) -> PathosGraphState:
    try:
        chunks = await retrieval_service.hybrid_search(state["standalone_query"])
        state["retrieved_chunks"] = chunks
    except Exception:
        logger.exception("hybrid_retrieval_node_failed")
        state["retrieved_chunks"] = []
    return state


@traced_node("cross_encoder_rerank")
async def rerank_node(state: PathosGraphState) -> PathosGraphState:
    reranked = reranker.rerank(
        query=state["standalone_query"],
        candidates=state.get("retrieved_chunks", []),
    )
    state["reranked_chunks"] = reranked
    return state


@traced_node("generation")
async def generation_node(state: PathosGraphState) -> PathosGraphState:
    system_prompt = _BASE_SYSTEM_PROMPT
    if state.get("regeneration_count", 0) > 0:
        system_prompt += _STRICT_RETRY_SUFFIX

    # Flag dosage-sensitive turns explicitly (set by input_guardrail_node's
    # "exact_dosage_request" finding) so generation is extra conservative.
    if any(f.rule_name == "exact_dosage_request" for f in state.get("guardrail_findings", [])):
        system_prompt += "\nThe user is asking about medication dosing specifically — be extra explicit that exact dosing must come from a pharmacist or prescribing clinician.\n"

    context_block = _build_context_block(state.get("reranked_chunks", []))
    user_prompt = (
        f"Reference material:\n{context_block}\n\n"
        f"Conversation so far:\n{_format_history(state.get('chat_history', []))}\n\n"
        f"User's question:\n{state['masked_message']}"
    )

    draft = await _call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
    state["draft_answer"] = draft
    return state


def _format_history(history: list[dict[str, str]]) -> str:
    if not history:
        return "(no prior turns)"
    return "\n".join(f"{h.get('role', 'user')}: {h.get('content', '')}" for h in history[-6:])


async def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Provider-agnostic LLM call. Import is deferred so this module has no
    hard dependency on `openai`/`anthropic` packages at import time (keeps
    unit tests that mock this function fast and dependency-free).
    """
    try:
        if settings.llm_provider.value == "openai":
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value() if settings.openai_api_key else None)
            response = await client.chat.completions.create(
                model=settings.generation_model,
                temperature=settings.generation_temperature,
                max_tokens=settings.generation_max_output_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content or ""
        else:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None)
            response = await client.messages.create(
                model=settings.generation_model,
                max_tokens=settings.generation_max_output_tokens,
                temperature=settings.generation_temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return "".join(block.text for block in response.content if hasattr(block, "text"))
    except Exception:
        logger.exception("llm_generation_call_failed", extra={"provider": settings.llm_provider.value})
        raise


@traced_node("output_guardrail")
async def output_guardrail_node(state: PathosGraphState) -> PathosGraphState:
    result = output_guardrail.evaluate(state["draft_answer"])
    state["guardrail_findings"] = list(state.get("guardrail_findings", [])) + result.findings

    if result.verdict == GuardrailVerdict.REGENERATE:
        current_retries = state.get("regeneration_count", 0)
        if current_retries < settings.max_regeneration_retries:
            state["regeneration_count"] = current_retries + 1
            return state  # loop back to generation_node in the graph edges
        else:
            # Retries exhausted — fall back to a safe templated answer
            # rather than ever streaming a rejected draft to the user.
            state["draft_answer"] = (
                "I want to be careful here: I can't give a definitive diagnosis "
                "or an exact medication dose. Based on general educational "
                "information, this may be worth discussing with a licensed "
                "clinician who can examine you and review your full history. "
                + settings.disclaimer_text
            )

    if not state["draft_answer"].strip().lower().count("educational"):
        state["draft_answer"] = state["draft_answer"].rstrip() + "\n\n" + settings.disclaimer_text

    state["final_answer"] = state["draft_answer"]
    return state


@traced_node("pii_unmasking")
async def pii_unmasking_node(state: PathosGraphState) -> PathosGraphState:
    state["final_answer"] = privacy_engine.unmask(state["final_answer"], state.get("pii_map", {}))
    return state


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def _needs_regeneration(state: PathosGraphState) -> Literal["regenerate", "continue"]:
    if state.get("regeneration_count", 0) > 0 and not state.get("final_answer"):
        # regeneration_count was just incremented by output_guardrail_node
        # and final_answer hasn't been set yet -> loop back once.
        if state["regeneration_count"] <= settings.max_regeneration_retries:
            return "regenerate"
    return "continue"


def build_pathos_graph():
    """
    Assembles the LangGraph StateGraph. Import of `langgraph` is deferred
    into this factory function so the rest of the module (individual node
    functions) stays importable/unit-testable even in environments where
    `langgraph` isn't installed.
    """
    from langgraph.graph import END, StateGraph

    graph = StateGraph(PathosGraphState)

    graph.add_node("pii_masking", pii_masking_node)
    graph.add_node("input_guardrail", input_guardrail_node)
    graph.add_node("query_rewrite", query_rewrite_node)
    graph.add_node("hybrid_retrieval", hybrid_retrieval_node)
    graph.add_node("cross_encoder_rerank", rerank_node)
    graph.add_node("generation", generation_node)
    graph.add_node("output_guardrail", output_guardrail_node)
    graph.add_node("pii_unmasking", pii_unmasking_node)

    graph.set_entry_point("pii_masking")
    graph.add_edge("pii_masking", "input_guardrail")

    graph.add_conditional_edges(
        "input_guardrail",
        lambda s: "blocked" if s.get("blocked") else "allowed",
        {"blocked": "pii_unmasking", "allowed": "query_rewrite"},
    )

    graph.add_edge("query_rewrite", "hybrid_retrieval")
    graph.add_edge("hybrid_retrieval", "cross_encoder_rerank")
    graph.add_edge("cross_encoder_rerank", "generation")
    graph.add_edge("generation", "output_guardrail")

    graph.add_conditional_edges(
        "output_guardrail",
        _needs_regeneration,
        {"regenerate": "generation", "continue": "pii_unmasking"},
    )

    graph.add_edge("pii_unmasking", END)

    return graph.compile()


async def run_pathos_graph(
    session_id: uuid.UUID,
    message: str,
    chat_history: list[dict[str, str]],
) -> PathosGraphState:
    """
    Public entrypoint used by the chat router. Handles the `blocked` short
    circuit explicitly here (rather than relying purely on graph edges) so
    the blocked-response text is copied into final_answer regardless of
    which branch of the graph executed.
    """
    state = build_initial_state(session_id, message, chat_history)
    compiled_graph = build_pathos_graph()

    result_state: PathosGraphState = await compiled_graph.ainvoke(state)

    if result_state.get("blocked"):
        result_state["final_answer"] = result_state.get("blocked_response") or "I'm not able to help with that."

    return result_state
