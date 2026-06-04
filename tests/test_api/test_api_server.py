"""
tests/test_api/test_api_server.py
──────────────────────────────────
Integration tests for the NOIA FastAPI server.
All ML pipeline components are replaced with lightweight mocks so the test
suite runs without GPU/model downloads.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── Patch heavy ML singletons before importing the app ────────────────────────

_mock_retriever = MagicMock()
_mock_reranker  = MagicMock()
_mock_engine    = MagicMock()
_mock_pipeline  = MagicMock()

_DUMMY_CHUNK = {
    "chunk_id": "abc123", "doc_id": "RB-0001", "source_type": "runbook",
    "source_file": "test.txt", "title": "BGP Runbook", "timestamp": "2024-01-01",
    "chunk_index": 0, "text": "BGP recovery procedure.", "score": 0.92,
}

_DUMMY_NOIA_RESPONSE = MagicMock()
_DUMMY_NOIA_RESPONSE.to_dict.return_value = {
    "mode":            "query",
    "query":           "test query",
    "response":        {"answer": "Mock answer.", "key_steps": [], "confidence": "HIGH", "caveats": ""},
    "groundedness":    {"groundedness_score": 0.91, "grounded_claims": 5,
                        "total_claims": 5, "review_required": False},
    "review_required": False,
    "response_time_ms":245.0,
    "llm_backend":     "openai",
    "model":           "mock-model",
    "context_sources": [
        {"source_num": 1, "doc_id": "RB-0001", "title": "BGP Runbook",
         "source_type": "runbook", "timestamp": "2024-01-01", "score": 0.92}
    ],
    "errors": [],
}

_mock_retriever.retrieve.return_value = [MagicMock(**_DUMMY_CHUNK)]
_mock_retriever.retrieve_by_source_types.return_value = [MagicMock(**_DUMMY_CHUNK)]
_mock_retriever.collection_stats.return_value = {
    "dense_chunk_count": 500, "sparse_chunk_count": 500,
    "hybrid_alpha": 0.55, "embedding_model": "all-MiniLM-L6-v2",
}
_mock_reranker.rerank.return_value  = [MagicMock(**_DUMMY_CHUNK)]
_mock_engine.answer_query.return_value           = _DUMMY_NOIA_RESPONSE
_mock_engine.analyse_rca.return_value            = _DUMMY_NOIA_RESPONSE
_mock_engine.summarise_incident.return_value     = _DUMMY_NOIA_RESPONSE
_mock_engine.recommend_fix.return_value          = _DUMMY_NOIA_RESPONSE
_mock_engine.generate_kb_article.return_value    = _DUMMY_NOIA_RESPONSE
_mock_pipeline.get_index_stats.return_value      = {
    "dense_chunk_count": 500, "sparse_chunk_count": 500,
    "collection": "noia_knowledge_base", "bm25_path": "data/bm25.pkl",
}


@pytest.fixture(scope="module")
def client():
    with (
        patch("noia.api.api_server.HybridRetriever", return_value=_mock_retriever),
        patch("noia.api.api_server.Reranker",         return_value=_mock_reranker),
        patch("noia.api.api_server.ReasoningEngine",  return_value=_mock_engine),
        patch("noia.api.api_server.IngestPipeline",   return_value=_mock_pipeline),
        patch("noia.api.api_server._init_audit_db",   return_value=MagicMock()),
    ):
        from noia.api.api_server import app  # noqa: PLC0415
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ── Helper: obtain auth token ─────────────────────────────────────────────────

def _get_token(client, username="noc_engineer_demo", password="noia_demo_2025") -> str:
    resp = client.post("/api/v1/auth/token",
                       data={"username": username, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


# ── Auth tests ────────────────────────────────────────────────────────────────

class TestAuth:
    def test_valid_login_returns_token(self, client):
        resp = client.post("/api/v1/auth/token",
                           data={"username": "noc_engineer_demo",
                                 "password": "noia_demo_2025"})
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0

    def test_invalid_credentials_returns_401(self, client):
        resp = client.post("/api/v1/auth/token",
                           data={"username": "noc_engineer_demo",
                                 "password": "wrong_password"})
        assert resp.status_code == 401

    def test_unknown_user_returns_401(self, client):
        resp = client.post("/api/v1/auth/token",
                           data={"username": "ghost", "password": "pass"})
        assert resp.status_code == 401

    def test_unauthenticated_query_returns_401(self, client):
        resp = client.post("/api/v1/query", json={"query": "BGP session drop"})
        assert resp.status_code == 401


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_body_structure(self, client):
        data = client.get("/api/v1/health").json()
        assert data["status"] == "healthy"
        assert "dense_chunks"   in data
        assert "sparse_chunks"  in data
        assert "llm_backend"    in data
        assert "embedding_model" in data


# ── TC-01: Query ──────────────────────────────────────────────────────────────

class TestQueryEndpoint:
    def test_valid_query_returns_200(self, client):
        token = _get_token(client)
        resp  = client.post(
            "/api/v1/query",
            json={"query": "How do I restore a BGP session after a route flap?"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_response_contains_required_fields(self, client):
        token = _get_token(client)
        data  = client.post(
            "/api/v1/query",
            json={"query": "BGP route flap recovery steps."},
            headers={"Authorization": f"Bearer {token}"},
        ).json()
        assert "response"         in data
        assert "groundedness"     in data
        assert "context_sources"  in data
        assert "review_required"  in data
        assert "response_time_ms" in data

    def test_short_query_returns_422(self, client):
        token = _get_token(client)
        resp  = client.post(
            "/api/v1/query",
            json={"query": "hi"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 422

    def test_read_only_can_query(self, client):
        token = _get_token(client, "readonly_demo", "readonly_2025")
        resp  = client.post(
            "/api/v1/query",
            json={"query": "What is the permitted downtime per month for 99.99% SLA?"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200


# ── TC-02: RCA ────────────────────────────────────────────────────────────────

class TestRCAEndpoint:
    def test_valid_rca_returns_200(self, client):
        token = _get_token(client)
        resp  = client.post(
            "/api/v1/rca",
            json={"incident_description":
                  "PE-03 Lagos BGP session dropped at 02:35 UTC. "
                  "Interface TenGigE0/0/1 shows 1200 CRC errors per second. "
                  "Maintenance CHG-00042 completed 6 hours prior."},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_read_only_cannot_access_rca(self, client):
        token = _get_token(client, "readonly_demo", "readonly_2025")
        resp  = client.post(
            "/api/v1/rca",
            json={"incident_description": "BGP session dropped on PE-01 Lagos region."},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403


# ── TC-03: Summarise ──────────────────────────────────────────────────────────

class TestSummariseEndpoint:
    def test_valid_summarise_returns_200(self, client):
        token = _get_token(client)
        resp  = client.post(
            "/api/v1/summarise",
            json={"incident_thread":
                  "INC-000045 opened 2024-03-12 01:15 UTC. P1 incident. "
                  "Node P-04 Nairobi. Optical power degradation LOS alarm triggered. "
                  "Engineer acknowledged 01:28. OTDR dispatched 02:00. "
                  "Fiber break confirmed at GPS coordinates. Crew arrived 04:30. "
                  "Fiber spliced successfully at 06:15. Service restored 06:45. "
                  "Total downtime 330 minutes. SLA breach confirmed."},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200


# ── TC-04: Recommend ──────────────────────────────────────────────────────────

class TestRecommendEndpoint:
    def test_valid_recommend_returns_200(self, client):
        token = _get_token(client)
        resp  = client.post(
            "/api/v1/recommend",
            json={"fault_description":
                  "CPU overload on PE-07 Cairo. CPU at 98% for 15 minutes. "
                  "BGP keepalives are being delayed."},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_read_only_cannot_recommend(self, client):
        token = _get_token(client, "readonly_demo", "readonly_2025")
        resp  = client.post(
            "/api/v1/recommend",
            json={"fault_description": "CPU overload on PE-07 Cairo at 98%."},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403


# ── TC-05: Generate Article ───────────────────────────────────────────────────

class TestGenerateArticleEndpoint:
    def test_valid_generate_returns_200(self, client):
        token = _get_token(client)
        resp  = client.post(
            "/api/v1/generate-article",
            json={"resolved_incident":
                  "INC-000088 resolved. BGP session drop on PE-01 caused by "
                  "misconfigured hold timer (3 seconds) introduced during software "
                  "upgrade rollback. Resolution: corrected timer to 90 seconds. "
                  "Prevention: add timer validation to upgrade checklist."},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
