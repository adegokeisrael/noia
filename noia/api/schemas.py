"""
noia/api/schemas.py
────────────────────
Pydantic v2 request and response schemas for the NOIA FastAPI server.

All API I/O is typed and validated at the boundary. Downstream modules
receive plain Python objects (dataclasses) rather than Pydantic models.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ── Shared models ─────────────────────────────────────────────────────────────

class SourceCitation(BaseModel):
    """Metadata for a single retrieved source document."""
    source_num:  int
    doc_id:      str
    title:       str
    source_type: str
    timestamp:   str
    score:       float


class GroundednessInfo(BaseModel):
    """Groundedness assessment summary."""
    groundedness_score: float
    grounded_claims:    int
    total_claims:       int
    review_required:    bool


class BaseResponse(BaseModel):
    """Common fields present in every NOIA API response."""
    mode:              str
    response:          dict[str, Any]
    groundedness:      GroundednessInfo
    review_required:   bool
    response_time_ms:  float
    llm_backend:       str
    model:             str
    context_sources:   list[SourceCitation]
    errors:            list[str] = Field(default_factory=list)


# ── Request models ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """
    TC-01: General NOC runbook / knowledge query.
    """
    query: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Natural-language query from the NOC engineer.",
        examples=["What are the steps to restore a BGP session after a route flap on a Nokia 7750 SR?"],
    )
    source_type_filter: Literal["runbook", "incident", "sla", "maintenance"] | None = Field(
        None,
        description="Restrict retrieval to a specific knowledge source type.",
    )
    top_k: int = Field(5, ge=1, le=20, description="Number of context chunks.")

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()


class RCARequest(BaseModel):
    """
    TC-02: Outage root cause analysis request.
    """
    incident_description: str = Field(
        ...,
        min_length=20,
        max_length=5000,
        description="Description of the outage: affected nodes, alarms, timeline.",
        examples=[
            "At 02:35 UTC, PE-03 in Lagos generated a BGP session drop alarm "
            "for peer 10.0.0.1. Interface TenGigE0/0/1 shows CRC errors. "
            "Maintenance window CHG-00042 was completed 6 hours earlier."
        ],
    )
    affected_nodes: list[str] = Field(
        default_factory=list,
        description="List of affected network node identifiers.",
    )
    alarm_codes: list[str] = Field(
        default_factory=list,
        description="List of active alarm codes from NMS.",
    )
    top_k: int = Field(5, ge=1, le=20)


class SummariseRequest(BaseModel):
    """
    TC-03: Incident thread summarisation.
    """
    incident_thread: str = Field(
        ...,
        min_length=50,
        max_length=20000,
        description="Full incident ticket thread text (multi-turn conversation, logs, notes).",
    )
    top_k: int = Field(5, ge=1, le=20)


class RecommendRequest(BaseModel):
    """
    TC-04: Fix recommendation with SLA awareness.
    """
    fault_description: str = Field(
        ...,
        min_length=10,
        max_length=3000,
        description="Description of the current fault or anomaly.",
    )
    affected_service: str | None = Field(
        None,
        description="Name of the affected service (for SLA lookup).",
    )
    sla_availability_commitment: float | None = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Customer SLA availability commitment in percent (e.g. 99.95).",
    )
    top_k: int = Field(5, ge=1, le=20)


class GenerateArticleRequest(BaseModel):
    """
    TC-05: KB article generation from a resolved incident.
    """
    resolved_incident: str = Field(
        ...,
        min_length=50,
        max_length=10000,
        description="Closed incident thread with resolution notes.",
    )
    top_k: int = Field(5, ge=1, le=20)


# ── Response models ───────────────────────────────────────────────────────────

class QueryResponse(BaseResponse):
    """TC-01 response."""
    pass


class RCAResponse(BaseResponse):
    """TC-02 response."""
    pass


class SummariseResponse(BaseResponse):
    """TC-03 response."""
    pass


class RecommendResponse(BaseResponse):
    """TC-04 response."""
    pass


class GenerateArticleResponse(BaseResponse):
    """TC-05 response."""
    pass


# ── Auth models ───────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=6, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    expires_in:   int


# ── Health / info models ──────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:           str
    version:          str
    dense_chunks:     int
    sparse_chunks:    int
    llm_backend:      str
    embedding_model:  str


class IndexStatsResponse(BaseModel):
    dense_chunk_count:  int
    sparse_chunk_count: int
    collection:         str
    bm25_path:          str
