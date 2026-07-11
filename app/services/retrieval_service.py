"""
Pathos AI — Retrieval Service
================================
Orchestrates hybrid search: dense vector similarity (Qdrant) fused with
BM25 sparse scoring, merged via weighted Reciprocal Rank Fusion (RRF), then
handed to the cross-encoder reranker. This is the "Vector Retrieval" node's
implementation, called from `app/engines/llm_graph.py`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import settings
from app.engines.embeddings import bm25_encoder, dense_embedder
from app.schemas import RetrievedChunk

logger = logging.getLogger("pathos_ai.retrieval")


@dataclass
class _InMemoryChunk:
    chunk_id: str
    document_title: str
    source_url: str | None
    text: str


class RetrievalService:
    """
    Wraps a Qdrant client for dense retrieval. Also keeps a lightweight
    in-memory mirror of chunk text for the BM25 sparse pass — in a full
    production deployment this mirror is replaced by Qdrant's native sparse
    vector index (see docs/architecture.md trade-offs section), but the
    RRF fusion logic below is identical either way.
    """

    def __init__(self) -> None:
        self._client = None
        self._chunks: dict[str, _InMemoryChunk] = {}
        self._bm25_fitted = False

    # -- Lifecycle -------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            from qdrant_client import AsyncQdrantClient

            self._client = AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
            )
        return self._client

    async def ensure_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        client = self._get_client()
        collections = await client.get_collections()
        existing = {c.name for c in collections.collections}
        if settings.qdrant_collection not in existing:
            await client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(size=settings.active_embedding_dimensions, distance=Distance.COSINE),
            )
            logger.info("qdrant_collection_created", extra={"collection": settings.qdrant_collection})

    async def index_documents(self, documents: list[_InMemoryChunk]) -> None:
        """Upserts documents into both the dense (Qdrant) and sparse (BM25) indices."""
        from qdrant_client.models import PointStruct

        if not documents:
            return

        await self.ensure_collection()
        client = self._get_client()

        texts = [d.text for d in documents]
        vectors = await dense_embedder.embed_documents(texts)

        points = [
            PointStruct(
                id=doc.chunk_id,
                vector=vector,
                payload={
                    "document_title": doc.document_title,
                    "source_url": doc.source_url,
                    "text": doc.text,
                },
            )
            for doc, vector in zip(documents, vectors)
        ]
        await client.upsert(collection_name=settings.qdrant_collection, points=points)

        for doc in documents:
            self._chunks[doc.chunk_id] = doc
        bm25_encoder.fit({c.chunk_id: c.text for c in self._chunks.values()})
        self._bm25_fitted = True

        logger.info("documents_indexed", extra={"count": len(documents)})

    # -- Retrieval -------------------------------------------------------------

    async def hybrid_search(self, query: str) -> list[RetrievedChunk]:
        """
        Runs dense + sparse retrieval concurrently, fuses with weighted RRF,
        and returns unified RetrievedChunk candidates for the reranker.
        """
        dense_hits = await self._dense_search(query)
        sparse_hits = self._sparse_search(query)

        return self._fuse(dense_hits, sparse_hits)

    async def _dense_search(self, query: str) -> list[RetrievedChunk]:
        try:
            client = self._get_client()
            query_vector = (await dense_embedder.embed_documents([query]))[0]
            results = await client.search(
                collection_name=settings.qdrant_collection,
                query_vector=query_vector,
                limit=settings.retrieval_top_k_dense,
            )
            return [
                RetrievedChunk(
                    chunk_id=str(hit.id),
                    document_title=hit.payload.get("document_title", "Untitled"),
                    source_url=hit.payload.get("source_url"),
                    text=hit.payload.get("text", ""),
                    dense_score=float(hit.score),
                )
                for hit in results
            ]
        except Exception:
            logger.exception("dense_search_failed")
            return []

    def _sparse_search(self, query: str) -> list[RetrievedChunk]:
        if not self._bm25_fitted:
            return []
        hits = bm25_encoder.score(query, top_k=settings.retrieval_top_k_sparse)
        results = []
        for chunk_id, score in hits:
            chunk = self._chunks.get(chunk_id)
            if not chunk:
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_title=chunk.document_title,
                    source_url=chunk.source_url,
                    text=chunk.text,
                    sparse_score=score,
                )
            )
        return results

    @staticmethod
    def _fuse(
        dense_hits: list[RetrievedChunk],
        sparse_hits: list[RetrievedChunk],
        rrf_k: int = 60,
    ) -> list[RetrievedChunk]:
        """
        Weighted Reciprocal Rank Fusion. Using RANK (not raw score) avoids
        the classic pitfall of dense cosine similarity and BM25 scores
        living on incompatible scales.
        """
        weight = settings.hybrid_dense_weight
        merged: dict[str, RetrievedChunk] = {}
        fused_scores: dict[str, float] = {}

        for rank, chunk in enumerate(dense_hits):
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + weight / (rrf_k + rank + 1)
            merged[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(sparse_hits):
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + (1 - weight) / (rrf_k + rank + 1)
            if chunk.chunk_id in merged:
                merged[chunk.chunk_id].sparse_score = chunk.sparse_score
            else:
                merged[chunk.chunk_id] = chunk

        ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)
        return [merged[cid] for cid in ranked_ids]


retrieval_service = RetrievalService()