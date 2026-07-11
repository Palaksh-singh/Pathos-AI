"""
Pathos AI — Embedding Engine
==============================
Wraps dense embedding generation (OpenAI/Anthropic-compatible embedding
models) and a sparse BM25 encoder, so `retrieval_service.py` can build a
true hybrid index instead of dense-only similarity search.

The BM25 implementation here is a compact, dependency-light port of the
standard Okapi BM25 scoring formula, run over a tokenized corpus held by
the retrieval service. In a production deployment with a very large corpus
this is swapped for Qdrant's native sparse vector support (uses the same
`SparseEncoder.encode()` interface), which is why the interface is kept
provider-agnostic here rather than hand-rolled into the retrieval service.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter

from app.config import settings

logger = logging.getLogger("pathos_ai.embeddings")

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


class DenseEmbedder:
    """
    Thin async wrapper around the configured embedding provider.
    Kept isolated behind this class so swapping providers (OpenAI ->
    Anthropic-compatible endpoint -> local sentence-transformers model)
    never touches calling code in retrieval_service.py.
    """

    def __init__(self, model: str | None = None, dimensions: int | None = None) -> None:
        self.model = model or settings.embedding_model
        self.dimensions = dimensions or settings.embedding_dimensions

    async def embed_query(self, text: str) -> list[float]:
        return await self._embed_batch([text])[0] if False else (await self._embed_batch([text]))[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed_batch(texts)

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Calls the configured embedding provider. Import is deferred so this
        module has zero hard dependency on the `openai` package when the
        engine isn't in use (e.g. during unit tests that mock this class).
        """
        try:
            if settings.llm_provider.value == "openai":
                from openai import AsyncOpenAI

                client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value() if settings.openai_api_key else None)
                response = await client.embeddings.create(model=self.model, input=texts)
                return [item.embedding for item in response.data]
            else:
                # Anthropic does not currently expose a first-party embeddings
                # endpoint; fall back to a configured OpenAI-compatible one,
                # or raise clearly so misconfiguration is obvious at runtime.
                raise RuntimeError(
                    "No embedding provider configured for the selected LLM_PROVIDER. "
                    "Set OPENAI_API_KEY (used for embeddings regardless of chat provider) "
                    "or configure a self-hosted embedding endpoint."
                )
        except Exception:
            logger.exception("dense_embedding_failed", extra={"model": self.model, "batch_size": len(texts)})
            raise


class BM25SparseEncoder:
    """
    Classic Okapi BM25 over an in-memory corpus. Fit once at index build
    time (or app startup, loading the same chunk set as the dense index),
    then scored per query at retrieval time.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_freqs: list[Counter[str]] = []
        self._doc_lengths: list[int] = []
        self._avg_doc_length: float = 0.0
        self._df: Counter[str] = Counter()  # document frequency per term
        self._n_docs: int = 0
        self._corpus_ids: list[str] = []

    def fit(self, documents: dict[str, str]) -> None:
        """documents: mapping of chunk_id -> raw text."""
        self._corpus_ids = list(documents.keys())
        self._doc_freqs = []
        self._doc_lengths = []
        self._df = Counter()

        for chunk_id in self._corpus_ids:
            tokens = tokenize(documents[chunk_id])
            counts = Counter(tokens)
            self._doc_freqs.append(counts)
            self._doc_lengths.append(len(tokens))
            for term in counts:
                self._df[term] += 1

        self._n_docs = len(self._corpus_ids)
        self._avg_doc_length = (sum(self._doc_lengths) / self._n_docs) if self._n_docs else 0.0
        logger.info("bm25_index_built", extra={"n_docs": self._n_docs})

    def score(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._n_docs == 0:
            return []

        query_terms = tokenize(query)
        scores = [0.0] * self._n_docs

        for term in query_terms:
            df = self._df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (self._n_docs - df + 0.5) / (df + 0.5))

            for i, doc_counts in enumerate(self._doc_freqs):
                freq = doc_counts.get(term, 0)
                if freq == 0:
                    continue
                doc_len = self._doc_lengths[i]
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / max(self._avg_doc_length, 1e-6))
                scores[i] += idf * (freq * (self.k1 + 1)) / max(denom, 1e-6)

        ranked = sorted(zip(self._corpus_ids, scores), key=lambda x: x[1], reverse=True)
        return [item for item in ranked if item[1] > 0][:top_k]


dense_embedder = DenseEmbedder()
bm25_encoder = BM25SparseEncoder()
