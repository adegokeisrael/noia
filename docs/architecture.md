# NOIA System Architecture

## Overview

NOIA is organised into four layers, each corresponding to a stage of the
AI-native RAG pipeline described in the FG-AINN submission (§4–§7).

```
┌─────────────────────────────────────────────────────────────────┐
│  KNOWLEDGE INGESTION LAYER                                      │
│  Runbooks · Incidents · SLA Docs · Maintenance Logs            │
│  ↓ corpus_builder.py  ↓ ingest_pipeline.py                     │
├─────────────────────────────────────────────────────────────────┤
│  PROCESSING & INDEXING LAYER                                    │
│  Text Chunking → Sentence-T5 Embedding → ChromaDB (dense)      │
│                                        → BM25 Index (sparse)   │
│                                        → KG Builder            │
├─────────────────────────────────────────────────────────────────┤
│  RETRIEVAL & REASONING LAYER                                    │
│  HybridRetriever (α-blend) → Reranker → ReasoningEngine        │
│                                       → GroundednessChecker    │
├─────────────────────────────────────────────────────────────────┤
│  NOC INTERFACE LAYER                                            │
│  FastAPI REST API (5 endpoints) → React Chat UI                │
└─────────────────────────────────────────────────────────────────┘
```

## Core Equations

**Hybrid blending score:**
```
score(q, d) = α · sim_dense(q, d)  +  (1 − α) · score_BM25_norm(q, d)
```

**Cosine similarity (dense):**
```
sim_dense(q, d) = ⟨q⃗, d⃗⟩ / (‖q⃗‖ · ‖d⃗‖)
```

**Groundedness score:**
```
G(r) = |{s ∈ S(r) : ∃d ∈ D : sim(s, d) ≥ θ}| / |S(r)|
```

Responses with G(r) < 0.70 are flagged REVIEW_REQUIRED.

## Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `corpus_builder.py` | Synthetic telecom document generation |
| `ingest_pipeline.py` | Chunking, embedding, ChromaDB + BM25 indexing |
| `hybrid_retriever.py` | α-blended dense+sparse retrieval |
| `reranker.py` | Cross-encoder reranking of top-K candidates |
| `prompt_templates.py` | Structured CoT templates per task mode |
| `reasoning_engine.py` | LLM prompt construction, inference, response parsing |
| `groundedness_checker.py` | Citation-based hallucination detection |
| `kg_builder.py` | Entity/relation extraction, topological RCA augmentation |
| `api_server.py` | FastAPI application, JWT auth, RBAC, audit logging |

## Data Flow (Online Query Mode)

```
Engineer Query
     │
     ▼
HybridRetriever.retrieve(query, top_k=20)
     ├─ ChromaDB ANN (dense cosine, top-20)
     └─ BM25 query (sparse, top-20)
           │ merge + α-blend normalised scores
           ▼
     Reranker.rerank(query, candidates, top_n=5)
           │ cross-encoder scoring
           ▼
     ReasoningEngine.reason(query, chunks, mode)
           │ structured prompt + LLM completion
           ▼
     GroundednessChecker.check(response, chunks)
           │ G(r) = grounded_claims / total_claims
           ▼
     NOIAResponse → API JSON → React UI
```

## Security

- JWT Bearer tokens (HS256, configurable expiry)
- Role-based access: `noc_engineer`, `team_lead`, `read_only`
- All queries logged to SQLite audit database
- Responses with G(r) < 0.70 marked REVIEW_REQUIRED in the UI

## Evaluation Metrics (RAGAS)

| Metric | Description | Target |
|--------|-------------|--------|
| Faithfulness | Response claims supported by context | ≥ 0.90 |
| Answer Relevancy | Response relevant to query | ≥ 0.85 |
| Context Precision | Retrieved chunks relevant to query | ≥ 0.85 |
| Context Recall | Reference answer found in context | ≥ 0.85 |
