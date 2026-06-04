"""
noia/api/api_server.py
───────────────────────
NOIA FastAPI application.

Exposes five operational endpoints (TC-01 through TC-05) plus authentication,
health check, and index statistics endpoints. All query endpoints are protected
by JWT Bearer authentication with role-based access control.

Endpoints
─────────
POST /api/v1/auth/token          → Obtain JWT access token
POST /api/v1/query               → TC-01: General NOC query
POST /api/v1/rca                 → TC-02: Outage root cause analysis
POST /api/v1/summarise           → TC-03: Incident summarisation
POST /api/v1/recommend           → TC-04: Fix recommendation
POST /api/v1/generate-article    → TC-05: KB article generation
GET  /api/v1/health              → System health and index stats
GET  /api/v1/stats               → Detailed index statistics (team_lead only)
GET  /api/v1/audit               → Audit log (team_lead only)

Run with::

    uvicorn noia.api.api_server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm

from config.settings import get_settings
from noia.api.auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    require_permission,
)
from noia.api.schemas import (
    GenerateArticleRequest,
    GenerateArticleResponse,
    HealthResponse,
    IndexStatsResponse,
    QueryRequest,
    QueryResponse,
    RCARequest,
    RCAResponse,
    RecommendRequest,
    RecommendResponse,
    SummariseRequest,
    SummariseResponse,
    TokenResponse,
)
from noia.ingestion.ingest_pipeline import IngestPipeline
from noia.reasoning.reasoning_engine import NOIAResponse, ReasoningEngine
from noia.retrieval.hybrid_retriever import HybridRetriever
from noia.retrieval.reranker import Reranker

settings = get_settings()
log      = structlog.get_logger()

# ── Application state (singletons) ────────────────────────────────────────────

class AppState:
    retriever:  HybridRetriever | None = None
    reranker:   Reranker | None        = None
    engine:     ReasoningEngine | None = None
    pipeline:   IngestPipeline | None  = None
    audit_conn: sqlite3.Connection | None = None


_state = AppState()


# ── Audit log ─────────────────────────────────────────────────────────────────

def _init_audit_db() -> sqlite3.Connection:
    settings.audit_db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.audit_db_path), check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT NOT NULL,
            username     TEXT,
            endpoint     TEXT,
            query_hash   TEXT,
            mode         TEXT,
            groundedness REAL,
            review_flag  INTEGER,
            latency_ms   REAL
        )
    """)
    conn.commit()
    return conn


def _log_query(
    username: str,
    endpoint: str,
    query: str,
    noia_response: NOIAResponse,
) -> None:
    if _state.audit_conn is None:
        return
    import hashlib  # noqa: PLC0415
    q_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
    _state.audit_conn.execute(
        "INSERT INTO audit_log VALUES (NULL,?,?,?,?,?,?,?,?)",
        (
            datetime.now(tz=timezone.utc).isoformat(),
            username,
            endpoint,
            q_hash,
            noia_response.mode,
            noia_response.groundedness.score,
            int(noia_response.groundedness.review_required),
            noia_response.response_time_ms,
        ),
    )
    _state.audit_conn.commit()


# ── Lifespan: initialise and teardown singletons ──────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("NOIA API starting up…")

    _state.retriever = HybridRetriever(
        chroma_persist_dir = str(settings.chroma_persist_dir),
        chroma_collection  = settings.chroma_collection_name,
        bm25_index_path    = str(settings.bm25_index_path),
        embedding_model    = settings.embedding_model,
        hybrid_alpha       = settings.hybrid_alpha,
        top_k              = settings.retrieval_top_k,
    )
    _state.reranker = Reranker(
        model_name = settings.reranker_model,
        top_n      = settings.rerank_top_n,
    )
    _state.engine = ReasoningEngine(
        llm_backend        = settings.llm_backend,
        local_model_path   = str(settings.local_model_path),
        openai_api_base    = settings.openai_api_base,
        openai_api_key     = settings.openai_api_key,
        openai_model       = settings.openai_model,
        embedding_model    = settings.embedding_model,
        groundedness_theta = settings.groundedness_threshold,
    )
    _state.pipeline = IngestPipeline(
        chroma_persist_dir = str(settings.chroma_persist_dir),
        chroma_collection  = settings.chroma_collection_name,
        bm25_index_path    = str(settings.bm25_index_path),
        embedding_model    = settings.embedding_model,
        chunk_size         = settings.chunk_size,
        chunk_overlap      = settings.chunk_overlap,
    )
    _state.audit_conn = _init_audit_db()

    log.info("NOIA API ready.")
    yield

    if _state.audit_conn:
        _state.audit_conn.close()
    log.info("NOIA API shut down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "NOIA — Network Operations Intelligence Assistant",
    description = (
        "AI-native RAG-based decision support system for telecom NOC teams.\n\n"
        "**ITU-T FG-AINN Build-a-thon 2025** | African Institute of Telecommunications Research"
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # tighten in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Request timing middleware ─────────────────────────────────────────────────

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = str(round((time.perf_counter() - t0) * 1000, 1))
    return response


# ── Helper: retrieve + rerank ─────────────────────────────────────────────────

def _retrieve_context(query: str, top_k: int, source_type: str | None = None):
    candidates = _state.retriever.retrieve(query, top_k=settings.retrieval_top_k,
                                           source_type=source_type)
    reranked   = _state.reranker.rerank(query, candidates, top_n=top_k)
    return reranked


def _noia_response_to_api(noia_resp: NOIAResponse, schema_cls):
    d = noia_resp.to_dict()
    return schema_cls(
        mode             = d["mode"],
        response         = d["response"],
        groundedness     = d["groundedness"],
        review_required  = d["review_required"],
        response_time_ms = d["response_time_ms"],
        llm_backend      = d["llm_backend"],
        model            = d["model"],
        context_sources  = d["context_sources"],
        errors           = d["errors"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

# ── Auth ──────────────────────────────────────────────────────────────────────

@app.post(
    "/api/v1/auth/token",
    response_model=TokenResponse,
    tags=["Authentication"],
    summary="Obtain a JWT access token",
)
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]) -> TokenResponse:
    """Authenticate with username + password and receive a Bearer token."""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(data={"sub": user["username"], "role": user["role"]})
    return TokenResponse(
        access_token=token,
        expires_in=settings.access_token_expire_minutes * 60,
    )


# ── TC-01: General NOC Query ──────────────────────────────────────────────────

@app.post(
    "/api/v1/query",
    response_model=QueryResponse,
    tags=["NOC Operations"],
    summary="TC-01 — General NOC runbook / knowledge query",
)
async def query_endpoint(
    request:  QueryRequest,
    user: Annotated[dict, Depends(require_permission("query"))],
) -> QueryResponse:
    """
    Answer a general NOC engineer question by retrieving relevant passages
    from runbooks, incident history, SLA documents, and maintenance logs.
    """
    chunks   = _retrieve_context(request.query, request.top_k, request.source_type_filter)
    noia_r   = _state.engine.answer_query(request.query, chunks)
    _log_query(user["username"], "query", request.query, noia_r)
    return _noia_response_to_api(noia_r, QueryResponse)


# ── TC-02: Root Cause Analysis ────────────────────────────────────────────────

@app.post(
    "/api/v1/rca",
    response_model=RCAResponse,
    tags=["NOC Operations"],
    summary="TC-02 — Outage root cause analysis",
)
async def rca_endpoint(
    request: RCARequest,
    user:    Annotated[dict, Depends(require_permission("rca"))],
) -> RCAResponse:
    """
    Analyse an outage description and retrieve supporting evidence from
    incident history and runbooks to identify probable root causes with
    confidence scores.
    """
    # For RCA, retrieve across both runbooks and incidents
    combined = _state.retriever.retrieve_by_source_types(
        request.incident_description,
        source_types=["incident", "runbook", "maintenance"],
        per_type_k=request.top_k,
    )
    reranked = _state.reranker.rerank(request.incident_description, combined, top_n=request.top_k)
    noia_r   = _state.engine.analyse_rca(request.incident_description, reranked)
    _log_query(user["username"], "rca", request.incident_description, noia_r)
    return _noia_response_to_api(noia_r, RCAResponse)


# ── TC-03: Incident Summarisation ────────────────────────────────────────────

@app.post(
    "/api/v1/summarise",
    response_model=SummariseResponse,
    tags=["NOC Operations"],
    summary="TC-03 — Incident thread summarisation",
)
async def summarise_endpoint(
    request: SummariseRequest,
    user:    Annotated[dict, Depends(require_permission("summarise"))],
) -> SummariseResponse:
    """
    Summarise a full incident ticket thread into a concise structured summary
    with timeline, root cause, SLA impact, and lessons learned.
    """
    chunks = _retrieve_context(request.incident_thread[:800], request.top_k)
    noia_r = _state.engine.summarise_incident(request.incident_thread, chunks)
    _log_query(user["username"], "summarise", request.incident_thread[:200], noia_r)
    return _noia_response_to_api(noia_r, SummariseResponse)


# ── TC-04: Fix Recommendation ─────────────────────────────────────────────────

@app.post(
    "/api/v1/recommend",
    response_model=RecommendResponse,
    tags=["NOC Operations"],
    summary="TC-04 — Fix recommendation with SLA awareness",
)
async def recommend_endpoint(
    request: RecommendRequest,
    user:    Annotated[dict, Depends(require_permission("recommend"))],
) -> RecommendResponse:
    """
    Generate ranked remediation actions for a described fault, with risk
    assessment, rollback plans, and SLA impact warnings.
    """
    combined = _state.retriever.retrieve_by_source_types(
        request.fault_description,
        source_types=["runbook", "incident", "sla"],
        per_type_k=request.top_k,
    )
    reranked = _state.reranker.rerank(request.fault_description, combined, top_n=request.top_k)
    noia_r   = _state.engine.recommend_fix(request.fault_description, reranked)
    _log_query(user["username"], "recommend", request.fault_description, noia_r)
    return _noia_response_to_api(noia_r, RecommendResponse)


# ── TC-05: KB Article Generation ─────────────────────────────────────────────

@app.post(
    "/api/v1/generate-article",
    response_model=GenerateArticleResponse,
    tags=["Knowledge Management"],
    summary="TC-05 — Automatic KB article generation from resolved incident",
)
async def generate_article_endpoint(
    request: GenerateArticleRequest,
    user:    Annotated[dict, Depends(require_permission("generate_article"))],
) -> GenerateArticleResponse:
    """
    Convert a resolved incident thread into a reusable knowledge base article
    with diagnostic steps, resolution procedure, and prevention guidance.
    """
    combined = _state.retriever.retrieve_by_source_types(
        request.resolved_incident[:800],
        source_types=["incident", "runbook"],
        per_type_k=request.top_k,
    )
    reranked = _state.reranker.rerank(request.resolved_incident[:800], combined, top_n=request.top_k)
    noia_r   = _state.engine.generate_kb_article(request.resolved_incident, reranked)
    _log_query(user["username"], "generate_article", request.resolved_incident[:200], noia_r)
    return _noia_response_to_api(noia_r, GenerateArticleResponse)


# ── Health check ──────────────────────────────────────────────────────────────

@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="System health and index statistics",
)
async def health_endpoint() -> HealthResponse:
    """
    Return system health status and current index statistics.
    Does not require authentication (suitable for load-balancer probes).
    """
    stats = _state.retriever.collection_stats() if _state.retriever else {}
    return HealthResponse(
        status          = "healthy",
        version         = "1.0.0",
        dense_chunks    = stats.get("dense_chunk_count", 0),
        sparse_chunks   = stats.get("sparse_chunk_count", 0),
        llm_backend     = settings.llm_backend,
        embedding_model = settings.embedding_model,
    )


# ── Index stats (team_lead only) ─────────────────────────────────────────────

@app.get(
    "/api/v1/stats",
    response_model=IndexStatsResponse,
    tags=["System"],
    summary="Detailed index statistics (team_lead role required)",
)
async def stats_endpoint(
    user: Annotated[dict, Depends(require_permission("stats"))],
) -> IndexStatsResponse:
    stats = _state.pipeline.get_index_stats() if _state.pipeline else {}
    return IndexStatsResponse(**stats)


# ── Audit log (team_lead only) ────────────────────────────────────────────────

@app.get(
    "/api/v1/audit",
    tags=["System"],
    summary="Query audit log (team_lead role required)",
)
async def audit_endpoint(
    user:  Annotated[dict, Depends(require_permission("audit"))],
    limit: int = 50,
) -> list[dict]:
    """Return the most recent ``limit`` audit log entries."""
    if _state.audit_conn is None:
        return []
    rows = _state.audit_conn.execute(
        "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    cols = ["id", "timestamp", "username", "endpoint", "query_hash",
            "mode", "groundedness", "review_flag", "latency_ms"]
    return [dict(zip(cols, row)) for row in rows]
