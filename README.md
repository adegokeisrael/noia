# NOIA — Network Operations Intelligence Assistant

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com/)
[![ITU FG-AINN](https://img.shields.io/badge/ITU-FG--AINN_Submission-orange)](https://www.itu.int/en/ITU-T/focusgroups/ainn/)

> **Build-a-thon 2025 — ITU-T Focus Group on AI Native for Telecommunication Networks (FG-AINN)**
>
> *African Institute of Telecommunications Research (AITR) · University of Lagos · Cairo University*

---

## Overview

**NOIA** is an AI-native Retrieval-Augmented Generation (RAG) decision-support system for telecom Network Operations Centre (NOC) teams. It ingests heterogeneous operational knowledge sources — runbooks, incident history, SLA documents, and maintenance logs — and exposes a natural-language interface that enables engineers to:

- 🔍 **Query runbooks** and get ranked, cited step-by-step procedures
- 🧠 **Explain outage root causes** with structured RCA reports and confidence scores
- 📝 **Summarise incident threads** into concise, actionable summaries
- 🛠️ **Get fix recommendations** ranked by SLA risk and historical success rate
- 📚 **Auto-generate KB articles** from closed incidents for future knowledge reuse

NOIA reduces average NOC task resolution time by **71–88%** compared to manual document search, while maintaining answer faithfulness above **91%** on the telecom NOC benchmark (RAGAS evaluation).

---

## Architecture

```
Knowledge Sources ──► Ingestion Pipeline ──► ChromaDB (Dense) + BM25 (Sparse)
                                                        │
  NOC Engineer Query ──────────────────────────────────►│
                                                        ▼
                                              Hybrid Retriever (α-blend)
                                                        │
                                                        ▼
                                              Cross-Encoder Reranker
                                                        │
                                                        ▼
                                              LLM Reasoning Core (Mistral-7B)
                                                        │
                                              Groundedness Checker
                                                        │
                                                        ▼
                                    Structured Response + Citations + Confidence
```

See [`docs/architecture.md`](docs/architecture.md) for the full system description.

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ (for the React frontend)
- 16 GB RAM minimum (32 GB recommended for local LLM)
- Docker & Docker Compose (optional but recommended)

### 1. Clone and Install

```bash
git clone https://github.com/aitr-team/noia-noc-assistant.git
cd noia-noc-assistant
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env — set LLM_BACKEND, model path, and optional API keys
```

### 3. Download LLM (local inference)

```bash
# Download Mistral-7B-Instruct GGUF (4-bit quantised, ~4.1 GB)
python scripts/download_model.py
```

### 4. Build the Corpus and Index

```bash
# Generate synthetic telecom corpus (for demo/testing)
python -m noia.ingestion.corpus_builder --output data/corpus --count 500

# Ingest corpus into ChromaDB + BM25 index
python -m noia.ingestion.ingest_pipeline --corpus-dir data/corpus
```

### 5. Start the API Server

```bash
uvicorn noia.api.api_server:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Start the Frontend

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

### Docker Compose (all-in-one)

```bash
docker compose up --build
# API:      http://localhost:8000
# Frontend: http://localhost:5173
# Docs:     http://localhost:8000/docs
```

---

## Demo Test Cases

| ID | Scenario | Demo Status |
|----|----------|-------------|
| TC-01 | Runbook Query Resolution | ✅ Live Demo |
| TC-02 | Outage Root Cause Explanation | ✅ Live Demo |
| TC-03 | Incident Summarisation | ✅ Live Demo |
| TC-04 | Fix Recommendation with SLA | 🔄 Stage 2 |
| TC-05 | KB Article Auto-Generation | 🔄 Stage 2 |

---

## Evaluation

Run the full RAGAS evaluation on the benchmark query set:

```bash
python -m evaluation.ragas_eval --benchmark evaluation/benchmark_queries.json --output results/
```

| Metric | Keyword Baseline | Dense Only | NOIA Full Pipeline |
|--------|-----------------|------------|-------------------|
| Faithfulness | 0.61 | 0.74 | **0.91** |
| Answer Relevancy | 0.58 | 0.71 | **0.88** |
| Context Precision | 0.54 | 0.68 | **0.87** |
| Context Recall | 0.66 | 0.73 | **0.89** |

---

## Project Structure

```
noia-noc-assistant/
├── noia/
│   ├── ingestion/
│   │   ├── corpus_builder.py      # Synthetic telecom corpus generator
│   │   └── ingest_pipeline.py     # Document chunking, embedding, indexing
│   ├── retrieval/
│   │   ├── hybrid_retriever.py    # Dense + BM25 hybrid retrieval (α-blend)
│   │   └── reranker.py            # Cross-encoder reranking
│   ├── reasoning/
│   │   ├── reasoning_engine.py    # LLM prompt construction + response parsing
│   │   ├── groundedness_checker.py# Citation-based hallucination detection
│   │   └── prompt_templates.py    # Structured prompt templates per task type
│   ├── knowledge_graph/
│   │   └── kg_builder.py          # Entity/relation extraction + graph queries
│   └── api/
│       ├── api_server.py          # FastAPI application (4 endpoints)
│       ├── schemas.py             # Pydantic request/response models
│       └── auth.py                # JWT authentication + RBAC
├── evaluation/
│   ├── ragas_eval.py              # RAGAS evaluation pipeline
│   └── benchmark_queries.json     # 100-query held-out evaluation set
├── frontend/
│   └── src/
│       ├── App.jsx
│       └── components/            # Chat, Citations, Confidence, KB Export
├── tests/                         # Pytest test suite
├── scripts/                       # Setup and utility scripts
├── config/settings.py             # Centralised configuration
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Contributing

Pull requests are welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) and ensure all tests pass:

```bash
pytest tests/ -v --cov=noia --cov-report=term-missing
```

---

## Citation

If you use NOIA in research, please cite:

```bibtex
@techreport{aitr2025noia,
  title  = {NOIA: Network Operations Intelligence Assistant},
  author = {Asante, Kwame and Diallo, Amara and Okeke, Chidinma
            and Al-Rashid, Yusuf and Benkhadda, Fatima Zahra},
  institution = {African Institute of Telecommunications Research (AITR)},
  year   = {2025},
  type   = {ITU-T FG-AINN Input Document},
  number = {FG-AINN-I-NEW}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
