"""
noia/reasoning/groundedness_checker.py
────────────────────────────────────────
Groundedness checker for NOIA LLM responses.

Implements the groundedness metric G(r) described in §4.d of the FG-AINN
submission:

    G(r) = |{s ∈ S(r) : ∃d ∈ D : sim(s, d) ≥ θ}| / |S(r)|

where:
  - S(r)  = set of atomic claims extracted from response r
  - D     = set of retrieved context chunks
  - θ     = semantic similarity threshold (default 0.78)
  - sim   = cosine similarity between sentence embeddings

A response with G(r) < threshold is flagged with REVIEW_REQUIRED, preventing
low-confidence outputs from being presented to NOC engineers without human
review.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

# Similarity threshold θ
DEFAULT_THETA: float = 0.78
# Minimum groundedness before flagging
DEFAULT_REVIEW_THRESHOLD: float = 0.70


# ── Embedding model loader ────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    logger.info("Loading groundedness embedding model: %s", model_name)
    return SentenceTransformer(model_name)


# ── Sentence splitter ─────────────────────────────────────────────────────────

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+"
)


def _split_into_claims(text: str) -> list[str]:
    """
    Split text into atomic claims (sentences).

    Strips JSON boilerplate, step numbers, and very short fragments (< 20 chars).
    """
    # Remove JSON keys and brackets
    text = re.sub(r'"[a-z_]+":\s*', "", text)
    text = re.sub(r"[{}\[\]]", " ", text)
    # Remove citation markers like [Source 1]
    text = re.sub(r"\[Source \d+\]", "", text)
    # Split on sentence boundaries
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in sentences if len(s.strip()) >= 20]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D numpy vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ── Result data class ─────────────────────────────────────────────────────────

@dataclass
class GroundednessResult:
    """Groundedness assessment for a single LLM response."""

    score: float                       # G(r) ∈ [0, 1]
    grounded_claims: int
    total_claims: int
    review_required: bool
    claim_scores: list[dict]           # per-claim detail

    def to_dict(self) -> dict:
        return {
            "groundedness_score": round(self.score, 4),
            "grounded_claims":    self.grounded_claims,
            "total_claims":       self.total_claims,
            "review_required":    self.review_required,
            "claim_detail":       self.claim_scores,
        }


# ── Groundedness checker ──────────────────────────────────────────────────────

class GroundednessChecker:
    """
    Checks whether LLM response claims are supported by retrieved context.

    Parameters
    ----------
    embedding_model : str
        HuggingFace sentence-transformers model for claim embedding.
    theta : float
        Cosine similarity threshold for a claim to be considered grounded.
    review_threshold : float
        G(r) below this value triggers REVIEW_REQUIRED flag.
    """

    def __init__(
        self,
        embedding_model: str  = "sentence-transformers/all-MiniLM-L6-v2",
        theta: float          = DEFAULT_THETA,
        review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
    ) -> None:
        self.embedding_model  = embedding_model
        self.theta            = theta
        self.review_threshold = review_threshold

    @property
    def _model(self):
        return _load_embedding_model(self.embedding_model)

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Return embedding matrix of shape (len(texts), dim)."""
        return self._model.encode(texts, batch_size=32,
                                  show_progress_bar=False,
                                  convert_to_numpy=True)

    def check(
        self,
        response_text: str,
        context_chunks: list[dict],
    ) -> GroundednessResult:
        """
        Compute the groundedness score G(r) for ``response_text``.

        Parameters
        ----------
        response_text : str
            Full text of the LLM response (JSON or Markdown, as returned by
            the reasoning engine).
        context_chunks : list[dict]
            Retrieved chunks, each with at least a 'text' key.

        Returns
        -------
        GroundednessResult
            Structured groundedness assessment.
        """
        claims = _split_into_claims(response_text)
        context_texts = [c.get("text", "") for c in context_chunks if c.get("text")]

        if not claims:
            logger.warning("No claims extracted from response; returning G(r)=1.0.")
            return GroundednessResult(
                score=1.0, grounded_claims=0, total_claims=0,
                review_required=False, claim_scores=[],
            )

        if not context_texts:
            logger.warning("No context chunks provided; returning G(r)=0.0.")
            return GroundednessResult(
                score=0.0, grounded_claims=0, total_claims=len(claims),
                review_required=True, claim_scores=[],
            )

        try:
            claim_embs   = self._embed(claims)            # (n_claims, dim)
            context_embs = self._embed(context_texts)     # (n_ctx, dim)
        except Exception as exc:  # noqa: BLE001
            logger.error("Embedding failed in groundedness check: %s", exc)
            return GroundednessResult(
                score=0.5, grounded_claims=0, total_claims=len(claims),
                review_required=True, claim_scores=[],
            )

        grounded   = 0
        claim_detail: list[dict] = []

        for i, (claim, c_emb) in enumerate(zip(claims, claim_embs)):
            max_sim  = 0.0
            best_src = -1
            for j, ctx_emb in enumerate(context_embs):
                sim = _cosine_similarity(c_emb, ctx_emb)
                if sim > max_sim:
                    max_sim  = sim
                    best_src = j

            is_grounded = max_sim >= self.theta
            if is_grounded:
                grounded += 1

            claim_detail.append({
                "claim":        claim[:120],
                "max_similarity": round(max_sim, 4),
                "best_source":  best_src + 1,  # 1-indexed for display
                "grounded":     is_grounded,
            })

        g_r = grounded / len(claims)
        review = g_r < self.review_threshold

        logger.info(
            "Groundedness G(r)=%.3f (%d/%d claims grounded, review=%s).",
            g_r, grounded, len(claims), review,
        )

        return GroundednessResult(
            score          = g_r,
            grounded_claims= grounded,
            total_claims   = len(claims),
            review_required= review,
            claim_scores   = claim_detail,
        )

    def check_json_response(
        self,
        response_obj: dict,
        context_chunks: list[dict],
        text_fields: list[str] | None = None,
    ) -> GroundednessResult:
        """
        Check groundedness for a structured JSON response by extracting
        text-bearing fields.

        Parameters
        ----------
        response_obj : dict
            Parsed JSON response from the reasoning engine.
        context_chunks : list[dict]
            Retrieved context chunks.
        text_fields : list[str] | None
            Specific top-level keys to extract text from.
            If None, all string values are concatenated.
        """
        if text_fields:
            parts = [str(response_obj.get(f, "")) for f in text_fields]
        else:
            parts = _extract_all_strings(response_obj)

        full_text = " ".join(parts)
        return self.check(full_text, context_chunks)


def _extract_all_strings(obj, depth: int = 0) -> list[str]:
    """Recursively extract all string values from a nested dict/list."""
    if depth > 5:
        return []
    results = []
    if isinstance(obj, str):
        results.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            results.extend(_extract_all_strings(v, depth + 1))
    elif isinstance(obj, list):
        for item in obj:
            results.extend(_extract_all_strings(item, depth + 1))
    return results
