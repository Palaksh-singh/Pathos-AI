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

    Two providers are supported out of the box:
      - "openai": calls OpenAI's embeddings endpoint (or any OpenAI-compatible
        endpoint via `OPENAI_BASE_URL`, e.g. a self-hosted embedding server).
      - "local": runs a small sentence-transformers model on-device — zero
        API cost, zero network dependency, useful when you don't have (or
        don't want to spend) LLM provider credits just for retrieval.
    """

    def __init__(self, model: str | None = None, dimensions: int | None = None) -> None:
        self.model = model or (
            settings.local_embedding_model if settings.embedding_provider == "local" else settings.embedding_model
        )
        self.dimensions = dimensions or settings.active_embedding_dimensions
        self._local_model = None  # lazy-loaded fastembed model

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed_batch([text]))[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed_batch(texts)

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if settings.embedding_provider == "local":
            return await self._embed_batch_local(texts)
        return await self._embed_batch_openai(texts)

    async def _embed_batch_openai(self, texts: list[str]) -> list[list[float]]:
        """
        Calls the configured OpenAI-compatible embedding provider. Import is
        deferred so this module has zero hard dependency on the `openai`
        package when the engine isn't in use (e.g. during unit tests that
        mock this class).
        """
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=settings.openai_api_key.get_secret_value() if settings.openai_api_key else None,
                base_url=settings.openai_base_url,  # None -> official OpenAI endpoint
            )
            response = await client.embeddings.create(model=self.model, input=texts)
            return [item.embedding for item in response.data]
        except Exception:
            logger.exception("dense_embedding_failed", extra={"model": self.model, "batch_size": len(texts)})
            raise

    async def _embed_batch_local(self, texts: list[str]) -> list[list[float]]:
        """
        Runs a local ONNX embedding model via fastembed in a background
        thread (the library is synchronous/CPU-bound, so this keeps the
        async event loop from blocking on model inference). Deliberately
        uses fastembed rather than sentence-transformers: fastembed has no
        PyTorch dependency, which avoids PyTorch's extremely deep nested
        package paths that exceed Windows' default 260-character path limit
        on some systems. Model weights are downloaded once on first use and
        cached locally — after that, this path makes zero network calls.
        """
        import asyncio

        def _encode() -> list[list[float]]:
            if self._local_model is None:
                from fastembed import TextEmbedding

                self._local_model = TextEmbedding(model_name=self.model)
                logger.info("local_embedding_model_loaded", extra={"model": self.model})

            vectors = list(self._local_model.embed(texts))
            return [vector.tolist() for vector in vectors]

        try:
            return await asyncio.to_thread(_encode)
        except Exception:
            logger.exception("local_embedding_failed", extra={"model": self.model, "batch_size": len(texts)})
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