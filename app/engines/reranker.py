"""
Pathos AI — Cross-Encoder Reranker
======================================
Hybrid retrieval (dense + BM25) is good at recall but poor at precision —
it returns plausible-looking chunks that aren't actually responsive to the
query. This module re-scores the merged candidate set with a cross-encoder
(FlashRank's `ms-marco-MiniLM-L-12-v2` by default) that jointly encodes the
query and each candidate, then keeps only the top `rerank_top_k` chunks
under a hard token budget — this is the mechanism that prevents "context
stuffing" (dumping every marginally-related chunk into the prompt, which
both dilutes generation quality and blows the token budget).
"""
from __future__ import annotations

import logging

from app.config import settings
from app.schemas import RetrievedChunk

logger = logging.getLogger("pathos_ai.reranker")

# Rough heuristic: ~4 characters per token for English clinical text.
_CHARS_PER_TOKEN_ESTIMATE = 4


class CrossEncoderReranker:
    """
    Lazily loads the FlashRank cross-encoder on first use (avoids paying
    model-load cost at import time / during tests that never call rerank()).
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.reranker_model
        self._ranker = None

    def _get_ranker(self):
        if self._ranker is None:
            try:
                from flashrank import Ranker

                self._ranker = Ranker(model_name=self.model_name)
                logger.info("flashrank_model_loaded", extra={"model": self.model_name})
            except Exception:
                logger.exception("flashrank_load_failed", extra={"model": self.model_name})
                raise
        return self._ranker

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int | None = None,
        max_context_tokens: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Scores every candidate against the query, sorts descending, then
        greedily fills a token budget so total prompt context never exceeds
        `max_context_tokens` — even if that means returning fewer than
        `top_k` chunks.
        """
        if not candidates:
            return []

        top_k = top_k or settings.rerank_top_k
        max_context_tokens = max_context_tokens or settings.max_context_tokens

        try:
            from flashrank import RerankRequest

            ranker = self._get_ranker()
            passages = [
                {"id": c.chunk_id, "text": c.text, "meta": {"idx": i}}
                for i, c in enumerate(candidates)
            ]
            request = RerankRequest(query=query, passages=passages)
            results = ranker.rerank(request)

            score_by_id = {r["id"]: float(r["score"]) for r in results}
            for c in candidates:
                c.rerank_score = score_by_id.get(c.chunk_id, 0.0)

        except Exception:
            # Graceful degradation: if the cross-encoder model can't be
            # loaded (e.g. offline dev environment), fall back to blending
            # the existing dense/sparse scores rather than hard-failing the
            # whole request.
            logger.warning("cross_encoder_unavailable_falling_back_to_hybrid_score")
            for c in candidates:
                c.rerank_score = (0.5 * c.dense_score) + (0.5 * c.sparse_score)

        ranked = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)

        selected: list[RetrievedChunk] = []
        used_tokens = 0
        for chunk in ranked:
            estimated_tokens = max(1, len(chunk.text) // _CHARS_PER_TOKEN_ESTIMATE)
            if used_tokens + estimated_tokens > max_context_tokens:
                continue
            selected.append(chunk)
            used_tokens += estimated_tokens
            if len(selected) >= top_k:
                break

        logger.info(
            "rerank_complete",
            extra={"candidates": len(candidates), "selected": len(selected), "estimated_tokens": used_tokens},
        )
        return selected


reranker = CrossEncoderReranker()
