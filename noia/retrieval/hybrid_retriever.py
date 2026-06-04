"""
noia/retrieval/hybrid_retriever.py
────────────────────────────────────
Hybrid retrieval module for NOIA.

Implements the blended dense–sparse retrieval strategy described in §4.a of
the FG-AINN submission:

    score(q, d) = α · sim_dense(q, d) + (1 − α) · score_BM25_norm(q, d)

where ``α`` (``hybrid_alpha``) is a configurable blend coefficient optimised on
a held-out development set (empirically ≈ 0.55 on telecom NOC query datasets).

Dense retrieval:  ChromaDB cosine ANN search using sentence-transformers.
Sparse retrieval: BM25Okapi (rank_bm25) over a persisted inverted index.

Both sub-systems return scores normalised to [0, 1] before blending to ensure
a meaningful weighted combination.

Usage::

    from noia.retrieval.hybrid_retriever import HybridRetriever
    retriever = HybridRetriever()
    results = retriever.retrieve("BGP session drop on PE-01 Lagos", top_k=5)
    for r in results:
        print(r.score, r.source_type, r.title)
        print(r.text[:200])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Result data class ─────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A single chunk returned by the hybrid retriever."""

    chunk_id:    str
    doc_id:      str
    source_type: str        # runbook | incident | sla | maintenance
    source_file: str
    title:       str
    timestamp:   str
    chunk_index: int
    total_chunks: int
    text:        str
    dense_score:  float = 0.0
    sparse_score: float = 0.0
    score:        float = 0.0   # final blended score

    def to_dict(self) -> dict:
        return {
            "chunk_id":    self.chunk_id,
            "doc_id":      self.doc_id,
            "source_type": self.source_type,
            "source_file": self.source_file,
            "title":       self.title,
            "timestamp":   self.timestamp,
            "chunk_index": self.chunk_index,
            "text":        self.text,
            "dense_score": round(self.dense_score, 4),
            "sparse_score":round(self.sparse_score, 4),
            "score":       round(self.score, 4),
        }


# ── Normalisation helpers ─────────────────────────────────────────────────────

def _min_max_normalise(values: list[float]) -> list[float]:
    """Normalise a list of floats to [0, 1] using min-max scaling."""
    if not values:
        return []
    min_v = min(values)
    max_v = max(values)
    span  = max_v - min_v
    if span == 0.0:
        return [1.0 if v > 0 else 0.0 for v in values]
    return [(v - min_v) / span for v in values]


# ── HybridRetriever ───────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Blended dense–sparse retriever for the NOIA knowledge base.

    Parameters
    ----------
    chroma_persist_dir : str
        ChromaDB persistence directory.
    chroma_collection : str
        Name of the ChromaDB collection.
    bm25_index_path : str | Path
        Path to the persisted BM25 index (pickle file).
    embedding_model : str
        HuggingFace sentence-transformers model identifier.
    hybrid_alpha : float
        Blend coefficient α ∈ [0, 1].
        1.0 = pure dense retrieval.
        0.0 = pure sparse retrieval.
    top_k : int
        Number of candidates to retrieve from each sub-system before merging.
    """

    def __init__(
        self,
        chroma_persist_dir: str  = "data/chroma",
        chroma_collection: str   = "noia_knowledge_base",
        bm25_index_path: str     = "data/bm25_index.pkl",
        embedding_model: str     = "sentence-transformers/all-MiniLM-L6-v2",
        hybrid_alpha: float      = 0.55,
        top_k: int               = 20,
    ) -> None:
        self.chroma_persist_dir = chroma_persist_dir
        self.chroma_collection  = chroma_collection
        self.bm25_index_path    = Path(bm25_index_path)
        self.embedding_model    = embedding_model
        self.alpha              = hybrid_alpha
        self.top_k              = top_k

        self._chroma_col  = None
        self._bm25        = None
        self._emb_model   = None

    # ── Lazy initialisers ─────────────────────────────────────────────────────

    def _get_chroma_collection(self):
        if self._chroma_col is None:
            import chromadb  # noqa: PLC0415
            from chromadb.utils.embedding_functions import (  # noqa: PLC0415
                SentenceTransformerEmbeddingFunction,
            )
            client = chromadb.PersistentClient(path=self.chroma_persist_dir)
            emb_fn = SentenceTransformerEmbeddingFunction(model_name=self.embedding_model)
            self._chroma_col = client.get_or_create_collection(
                name=self.chroma_collection,
                embedding_function=emb_fn,
                metadata={"hnsw:space": "cosine"},
            )
        return self._chroma_col

    def _get_bm25(self):
        if self._bm25 is None:
            from noia.ingestion.ingest_pipeline import BM25Index  # noqa: PLC0415
            self._bm25 = BM25Index.load(self.bm25_index_path)
        return self._bm25

    # ── Sub-system queries ────────────────────────────────────────────────────

    def _dense_query(
        self,
        query: str,
        top_k: int,
        source_type_filter: str | None = None,
    ) -> dict[str, RetrievedChunk]:
        """
        Query ChromaDB for the top-k dense nearest neighbours.

        Returns a dict mapping chunk_id → RetrievedChunk.
        """
        col = self._get_chroma_collection()
        where = {"source_type": source_type_filter} if source_type_filter else None
        kwargs: dict = dict(
            query_texts=[query],
            n_results=min(top_k, col.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        results = col.query(**kwargs)
        chunks: dict[str, RetrievedChunk] = {}

        if not results["ids"] or not results["ids"][0]:
            return chunks

        ids       = results["ids"][0]
        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]

        # ChromaDB cosine distance ∈ [0, 2]; convert to similarity ∈ [-1, 1]
        # then shift to [0, 1]:  sim = 1 - distance/2
        for chunk_id, text, meta, dist in zip(ids, docs, metas, distances):
            dense_score = max(0.0, 1.0 - dist / 2.0)
            chunks[chunk_id] = RetrievedChunk(
                chunk_id    = chunk_id,
                doc_id      = meta.get("doc_id", ""),
                source_type = meta.get("source_type", ""),
                source_file = meta.get("source_file", ""),
                title       = meta.get("title", ""),
                timestamp   = meta.get("timestamp", ""),
                chunk_index = int(meta.get("chunk_index", 0)),
                total_chunks= int(meta.get("total_chunks", 1)),
                text        = text,
                dense_score = dense_score,
            )
        return chunks

    def _sparse_query(
        self,
        query: str,
        top_k: int,
        source_type_filter: str | None = None,
    ) -> dict[str, float]:
        """
        Query the BM25 index for the top-k sparse matches.

        Returns a dict mapping chunk_id → raw BM25 score.
        """
        bm25 = self._get_bm25()
        results = bm25.query(query, top_k=top_k)
        if source_type_filter:
            results = [
                r for r in results
                if r["metadata"].get("source_type") == source_type_filter
            ]
        return {r["chunk_id"]: r["score"] for r in results}

    # ── Merge & blend ─────────────────────────────────────────────────────────

    def _merge_results(
        self,
        dense_chunks: dict[str, RetrievedChunk],
        sparse_scores: dict[str, float],
    ) -> list[RetrievedChunk]:
        """
        Merge dense and sparse results, compute blended scores.

        Scores from each modality are independently min-max normalised to
        [0, 1] before blending:

            score(q, d) = α · norm_dense(d) + (1 − α) · norm_sparse(d)
        """
        all_chunk_ids = set(dense_chunks.keys()) | set(sparse_scores.keys())

        # Normalise dense scores
        dense_raw  = {cid: dense_chunks[cid].dense_score
                      for cid in dense_chunks}
        dense_norm = dict(zip(dense_raw.keys(),
                               _min_max_normalise(list(dense_raw.values()))))

        # Normalise sparse scores
        sparse_raw  = sparse_scores
        sparse_norm = dict(zip(sparse_raw.keys(),
                                _min_max_normalise(list(sparse_raw.values()))))

        merged: list[RetrievedChunk] = []

        for cid in all_chunk_ids:
            if cid in dense_chunks:
                chunk = dense_chunks[cid]
            else:
                # Chunk only found in sparse — fetch text from ChromaDB
                col = self._get_chroma_collection()
                raw = col.get(ids=[cid], include=["documents", "metadatas"])
                if not raw["ids"]:
                    continue
                meta = raw["metadatas"][0]
                chunk = RetrievedChunk(
                    chunk_id    = cid,
                    doc_id      = meta.get("doc_id", ""),
                    source_type = meta.get("source_type", ""),
                    source_file = meta.get("source_file", ""),
                    title       = meta.get("title", ""),
                    timestamp   = meta.get("timestamp", ""),
                    chunk_index = int(meta.get("chunk_index", 0)),
                    total_chunks= int(meta.get("total_chunks", 1)),
                    text        = raw["documents"][0],
                )

            d_score = dense_norm.get(cid, 0.0)
            s_score = sparse_norm.get(cid, 0.0)
            chunk.dense_score  = d_score
            chunk.sparse_score = s_score
            chunk.score        = self.alpha * d_score + (1.0 - self.alpha) * s_score
            merged.append(chunk)

        merged.sort(key=lambda c: c.score, reverse=True)
        return merged

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        source_type: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks for ``query``.

        Parameters
        ----------
        query : str
            Natural-language query from the NOC engineer.
        top_k : int | None
            Number of results to return. Defaults to ``self.top_k``.
        source_type : str | None
            Optional filter: 'runbook' | 'incident' | 'sla' | 'maintenance'.

        Returns
        -------
        list[RetrievedChunk]
            Ranked list of retrieved chunks, highest score first.
        """
        k = top_k or self.top_k
        logger.debug("Hybrid retrieve: query=%r, top_k=%d, filter=%s", query, k, source_type)

        dense_chunks  = self._dense_query(query, k, source_type)
        sparse_scores = self._sparse_query(query, k, source_type)

        merged = self._merge_results(dense_chunks, sparse_scores)
        top    = merged[:k]

        logger.info(
            "Retrieved %d chunks (dense=%d, sparse=%d, merged=%d).",
            len(top), len(dense_chunks), len(sparse_scores), len(merged),
        )
        return top

    def retrieve_by_source_types(
        self,
        query: str,
        source_types: list[str],
        per_type_k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Run separate hybrid retrievals per source type and interleave results.

        Useful for queries that explicitly need coverage across multiple
        knowledge source categories (e.g. RCA queries combining runbooks + incidents).

        Parameters
        ----------
        query : str
            Engineer query.
        source_types : list[str]
            Knowledge source types to query.
        per_type_k : int
            Chunks to retrieve per type before interleaving.

        Returns
        -------
        list[RetrievedChunk]
            Interleaved, deduplicated results.
        """
        seen: set[str] = set()
        interleaved: list[RetrievedChunk] = []

        buckets = [self.retrieve(query, top_k=per_type_k, source_type=st)
                   for st in source_types]

        # Round-robin interleave
        max_len = max((len(b) for b in buckets), default=0)
        for i in range(max_len):
            for bucket in buckets:
                if i < len(bucket):
                    chunk = bucket[i]
                    if chunk.chunk_id not in seen:
                        seen.add(chunk.chunk_id)
                        interleaved.append(chunk)

        return interleaved

    def collection_stats(self) -> dict:
        """Return basic statistics about the underlying indices."""
        col  = self._get_chroma_collection()
        bm25 = self._get_bm25()
        return {
            "dense_chunk_count":  col.count(),
            "sparse_chunk_count": len(bm25.chunk_ids),
            "hybrid_alpha":       self.alpha,
            "embedding_model":    self.embedding_model,
        }
