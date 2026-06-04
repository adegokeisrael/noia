"""
tests/test_reasoning/test_groundedness.py
──────────────────────────────────────────
Unit tests for groundedness checking utilities.
Embedding model is mocked with deterministic vectors.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from noia.reasoning.groundedness_checker import (
    GroundednessChecker,
    _cosine_similarity,
    _extract_all_strings,
    _split_into_claims,
)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 0.5, 0.25])
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 1.0])
        assert _cosine_similarity(a, b) == 0.0


class TestSplitIntoClaims:
    def test_basic_sentence_split(self):
        text = "The BGP session dropped. The hold timer expired. The interface was down."
        claims = _split_into_claims(text)
        assert len(claims) >= 2

    def test_short_fragments_excluded(self):
        text = "Yes. No. The router interface TenGigE0/0/1 experienced CRC errors at 1200 pps."
        claims = _split_into_claims(text)
        # Short "Yes" and "No" should be filtered
        assert all(len(c) >= 20 for c in claims)

    def test_empty_text_returns_empty(self):
        assert _split_into_claims("") == []

    def test_json_boilerplate_stripped(self):
        text = '"answer": "The BGP session was restored after a soft clear."'
        claims = _split_into_claims(text)
        assert any("BGP" in c for c in claims)


class TestExtractAllStrings:
    def test_flat_dict(self):
        obj = {"a": "hello", "b": "world"}
        result = _extract_all_strings(obj)
        assert "hello" in result
        assert "world" in result

    def test_nested_dict(self):
        obj = {"outer": {"inner": "deep_value"}}
        result = _extract_all_strings(obj)
        assert "deep_value" in result

    def test_list_of_strings(self):
        obj = ["step one", "step two", "step three"]
        result = _extract_all_strings(obj)
        assert result == ["step one", "step two", "step three"]

    def test_non_string_values_ignored(self):
        obj = {"count": 42, "flag": True, "text": "relevant"}
        result = _extract_all_strings(obj)
        assert "relevant" in result
        assert 42 not in result


class TestGroundednessChecker:
    """Use mocked embeddings for deterministic tests."""

    def _make_checker(self) -> GroundednessChecker:
        checker = GroundednessChecker(
            embedding_model="mock-model",
            theta=0.78,
            review_threshold=0.70,
        )
        return checker

    def _patch_embed(self, checker, claim_vecs, context_vecs):
        """Patch _embed to return deterministic vectors."""
        call_count = [0]
        matrices = [np.array(claim_vecs), np.array(context_vecs)]

        def fake_embed(texts):
            result = matrices[call_count[0] % 2]
            call_count[0] += 1
            return result

        checker._embed = fake_embed
        return checker

    def test_fully_grounded_response(self):
        checker = self._make_checker()
        # All claims perfectly match context (cosine sim = 1.0)
        v = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        checker._embed = lambda texts: v[:len(texts)]
        result = checker.check(
            "BGP was restored. The session is up. The procedure was followed.",
            [{"text": "BGP session restored successfully."}],
        )
        assert result.score == pytest.approx(1.0, abs=0.05)
        assert result.review_required is False

    def test_ungrounded_response_triggers_review(self):
        checker = self._make_checker()
        # Claims orthogonal to context → sim = 0.0
        claim_vecs   = np.array([[1.0, 0.0], [1.0, 0.0]])
        context_vecs = np.array([[0.0, 1.0]])
        call_count   = [0]

        def fake_embed(texts):
            idx = call_count[0]
            call_count[0] += 1
            if idx == 0:
                return claim_vecs[:len(texts)]
            return context_vecs[:len(texts)]

        checker._embed = fake_embed
        result = checker.check(
            "The router experienced a critical fault. System logged multiple errors.",
            [{"text": "unrelated context about something else entirely."}],
        )
        assert result.review_required is True

    def test_no_claims_extracted_returns_perfect_score(self):
        checker = self._make_checker()
        result = checker.check("OK.", [{"text": "context"}])
        assert result.score == 1.0
        assert result.review_required is False

    def test_no_context_returns_zero(self):
        checker = self._make_checker()
        result = checker.check("The BGP session was restored after a soft clear.", [])
        assert result.score == 0.0
        assert result.review_required is True

    def test_result_to_dict_structure(self):
        checker = self._make_checker()
        # Patch embed to return matching vectors
        v = np.array([[1.0, 0.0]] * 5)
        checker._embed = lambda texts: v[:len(texts)]
        result = checker.check(
            "The BGP session was restored. The procedure was executed correctly.",
            [{"text": "BGP session restored using soft clear procedure."}],
        )
        d = result.to_dict()
        assert "groundedness_score" in d
        assert "grounded_claims"   in d
        assert "total_claims"      in d
        assert "review_required"   in d
        assert "claim_detail"      in d
