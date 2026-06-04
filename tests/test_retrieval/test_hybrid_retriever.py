"""
tests/test_retrieval/test_hybrid_retriever.py
──────────────────────────────────────────────
Unit tests for hybrid retrieval utilities (normalisation, merge logic).
ChromaDB and sentence-transformers are mocked.
"""

from __future__ import annotations

import pytest

from noia.retrieval.hybrid_retriever import (
    RetrievedChunk,
    _min_max_normalise,
    HybridRetriever,
)


def _make_chunk(chunk_id, text, source_type="runbook", score=0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, doc_id=f"DOC-{chunk_id}", source_type=source_type,
        source_file="test.txt", title="Test", timestamp="2024-01-01",
        chunk_index=0, total_chunks=1, text=text,
        dense_score=score, sparse_score=score, score=score,
    )


class TestMinMaxNormalise:
    def test_uniform_values_map_to_one(self):
        result = _min_max_normalise([5.0, 5.0, 5.0])
        assert all(v == 1.0 for v in result)

    def test_range_maps_to_zero_one(self):
        result = _min_max_normalise([0.0, 0.5, 1.0])
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.5)
        assert result[2] == pytest.approx(1.0)

    def test_empty_returns_empty(self):
        assert _min_max_normalise([]) == []

    def test_single_nonzero_maps_to_one(self):
        result = _min_max_normalise([7.3])
        assert result[0] == 1.0

    def test_all_zeros_map_to_zero(self):
        result = _min_max_normalise([0.0, 0.0, 0.0])
        assert all(v == 0.0 for v in result)


class TestRetrievedChunkToDict:
    def test_to_dict_contains_required_keys(self):
        chunk = _make_chunk("c1", "BGP session drop", score=0.85)
        d = chunk.to_dict()
        for key in ["chunk_id", "doc_id", "source_type", "title", "text",
                    "dense_score", "sparse_score", "score"]:
            assert key in d

    def test_scores_rounded_to_4dp(self):
        chunk = _make_chunk("c1", "text", score=0.123456789)
        d = chunk.to_dict()
        assert d["score"] == pytest.approx(0.1235, abs=1e-4)


class TestHybridRetrieverMerge:
    """Test the _merge_results logic in isolation."""

    def _make_retriever(self) -> HybridRetriever:
        r = HybridRetriever.__new__(HybridRetriever)
        r.alpha = 0.55
        r.top_k = 20
        r._chroma_col = None
        r._bm25       = None
        r._emb_model  = None
        return r

    def test_merge_dense_only(self):
        retriever = self._make_retriever()
        dense = {
            "c1": _make_chunk("c1", "text one",   score=0.9),
            "c2": _make_chunk("c2", "text two",   score=0.6),
            "c3": _make_chunk("c3", "text three", score=0.3),
        }
        merged = retriever._merge_results(dense, {})
        assert len(merged) == 3
        # Should be sorted descending
        assert merged[0].chunk_id == "c1"
        assert merged[-1].chunk_id == "c3"

    def test_merge_preserves_all_chunk_ids(self):
        retriever = self._make_retriever()
        dense = {
            "c1": _make_chunk("c1", "BGP runbook", score=0.9),
            "c2": _make_chunk("c2", "OSPF incident", score=0.5),
        }
        sparse = {"c3": 8.0, "c2": 5.0}

        # Patch the chroma get call for sparse-only chunk c3
        import unittest.mock as mock
        with mock.patch.object(retriever, "_get_chroma_collection") as mock_col:
            mock_col.return_value.get.return_value = {
                "ids": ["c3"],
                "documents": ["spare chunk text"],
                "metadatas": [{"doc_id": "D3", "source_type": "incident",
                               "source_file": "f", "title": "T",
                               "timestamp": "2024", "chunk_index": 0, "total_chunks": 1}],
            }
            merged = retriever._merge_results(dense, sparse)

        ids = {c.chunk_id for c in merged}
        assert "c1" in ids
        assert "c2" in ids
        assert "c3" in ids

    def test_blend_score_uses_alpha(self):
        retriever = self._make_retriever()
        retriever.alpha = 1.0  # pure dense
        dense = {"c1": _make_chunk("c1", "text", score=0.8)}
        merged = retriever._merge_results(dense, {})
        # dense_norm of single value = 1.0; score = 1.0 * 1.0 + 0.0 * 0.0 = 1.0
        assert merged[0].score == pytest.approx(1.0)

    def test_blend_score_alpha_zero_is_pure_sparse(self):
        retriever = self._make_retriever()
        retriever.alpha = 0.0  # pure sparse
        dense = {
            "c1": _make_chunk("c1", "text A", score=0.95),
            "c2": _make_chunk("c2", "text B", score=0.10),
        }
        sparse = {"c1": 2.0, "c2": 8.0}
        merged = retriever._merge_results(dense, sparse)
        # With alpha=0, sparse dominates: c2 (score 8.0 > 2.0) should rank first
        assert merged[0].chunk_id == "c2"
