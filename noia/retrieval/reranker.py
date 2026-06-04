"""
noia/retrieval/reranker.py
───────────────────────────
Cross-encoder reranker for NOIA.

After hybrid retrieval produces up to K=20 candidates, the reranker applies a
cross-encoder model (jointly encoding query + passage) to produce a refined
relevance score for each candidate. The top-N (default 5) are selected as the
final context passed to the LLM reasoning core.

Cross-encoders are substantially more accurate than bi-encoders at the cost of
O(K) forward passes per query, which is acceptable for K ≤ 20 on modern hardware.

References
----------
- Nogueira & Cho (2019), "Passage Re-ranking with BERT"
- Humeau et al. (2020), "Poly-encoders: Transformer architectures for fast and
  accurate multi-sentence scoring"
"""

from __future__ import annotations

import logging
from functools import lru_cache

from noia.retrieval.hybrid_retriever import RetrievedChunk

logger = logging.getLogger(__name__)


# ── Model loader (cached singleton) ──────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_cross_encoder(model_name: str):
    """Load and cache the cross-encoder model."""
    from sentence_transformers import CrossEncoder  # noqa: PLC0415
    logger.info("Loading cross-encoder model: %s", model_name)
    return CrossEncoder(model_name, max_length=512)


# ── Reranker ──────────────────────────────────────────────────────────────────

class Reranker:
    """
    Cross-encoder reranker for re-scoring hybrid retrieval candidates.

    Parameters
    ----------
    model_name : str
        HuggingFace cross-encoder model identifier.
    top_n : int
        Number of top candidates to return after reranking.
    score_threshold : float | None
        If set, discard candidates whose rerank score falls below this value.
    """

    def __init__(
        self,
        model_name: str  = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int       = 5,
        score_threshold: float | None = None,
    ) -> None:
        self.model_name      = model_name
        self.top_n           = top_n
        self.score_threshold = score_threshold

    @property
    def _model(self):
        return _load_cross_encoder(self.model_name)

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Rerank ``chunks`` with respect to ``query`` using a cross-encoder.

        Each chunk in the returned list has an updated ``.score`` attribute
        representing the cross-encoder relevance logit (not normalised to [0,1]
        since the ranking order is what matters downstream).

        Parameters
        ----------
        query : str
            Original engineer query.
        chunks : list[RetrievedChunk]
            Candidate chunks from hybrid retrieval.
        top_n : int | None
            Override the instance-level ``top_n``.

        Returns
        -------
        list[RetrievedChunk]
            Top-N chunks sorted by cross-encoder score, descending.
        """
        n = top_n or self.top_n
        if not chunks:
            return []

        pairs = [(query, chunk.text) for chunk in chunks]

        try:
            scores: list[float] = self._model.predict(pairs, show_progress_bar=False).tolist()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Cross-encoder prediction failed (%s). Falling back to hybrid scores.", exc
            )
            scores = [c.score for c in chunks]

        # Attach rerank scores
        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)

        ranked = sorted(chunks, key=lambda c: c.score, reverse=True)

        if self.score_threshold is not None:
            ranked = [c for c in ranked if c.score >= self.score_threshold]

        result = ranked[:n]
        logger.info(
            "Reranked %d → %d chunks (top score=%.4f).",
            len(chunks), len(result), result[0].score if result else 0.0,
        )
        return result

    def rerank_with_diversity(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int | None = None,
        diversity_penalty: float = 0.1,
    ) -> list[RetrievedChunk]:
        """
        Rerank with a Maximal Marginal Relevance (MMR)-style diversity penalty.

        Penalises candidates that originate from the same source document as
        an already-selected chunk, encouraging the context window to include
        evidence from multiple source documents.

        Parameters
        ----------
        query : str
            Engineer query.
        chunks : list[RetrievedChunk]
            Candidate pool.
        top_n : int | None
            Number of results.
        diversity_penalty : float
            Score reduction applied to additional chunks from the same doc_id.

        Returns
        -------
        list[RetrievedChunk]
            Diverse top-N chunks.
        """
        n = top_n or self.top_n

        # First, standard rerank to get relevance scores
        reranked = self.rerank(query, chunks, top_n=len(chunks))
        if not reranked:
            return []

        selected: list[RetrievedChunk] = []
        selected_doc_ids: set[str] = set()

        for chunk in reranked:
            penalty = diversity_penalty if chunk.doc_id in selected_doc_ids else 0.0
            chunk.score -= penalty
            selected.append(chunk)
            selected_doc_ids.add(chunk.doc_id)
            if len(selected) == n:
                break

        # Re-sort after penalty application
        selected.sort(key=lambda c: c.score, reverse=True)
        return selected
