#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_demo.sh — NOIA Build-a-thon 2025 demo launcher
# Builds corpus, ingests it, and starts the API server.
# Usage: ./scripts/run_demo.sh [--openai]
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

BLUE='\033[0;34m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${BLUE}══════════════════════════════════════════${NC}"
echo -e "${BLUE}  NOIA — Network Operations Intelligence Assistant${NC}"
echo -e "${BLUE}  ITU-T FG-AINN Build-a-thon 2025${NC}"
echo -e "${BLUE}══════════════════════════════════════════${NC}\n"

# ── Check Python ──────────────────────────────────────────────────────────────
python_version=$(python --version 2>&1)
echo -e "${GREEN}✓ Python:${NC} $python_version"

# ── Check / create .env ───────────────────────────────────────────────────────
if [ ! -f .env ]; then
    cp .env.example .env
    echo -e "${YELLOW}⚠  Created .env from .env.example — edit before production use.${NC}"
fi

# ── Parse flags ───────────────────────────────────────────────────────────────
USE_OPENAI=false
for arg in "$@"; do
    case $arg in --openai) USE_OPENAI=true ;; esac
done

if $USE_OPENAI; then
    sed -i 's/^LLM_BACKEND=.*/LLM_BACKEND=openai/' .env
    echo -e "${GREEN}✓ LLM backend:${NC} OpenAI-compatible API"
else
    echo -e "${GREEN}✓ LLM backend:${NC} Local (llama.cpp)"
fi

# ── Create directories ────────────────────────────────────────────────────────
mkdir -p data/corpus data/chroma models results
echo -e "${GREEN}✓ Directories ready.${NC}"

# ── Step 1: Generate corpus (if not present) ──────────────────────────────────
if [ -z "$(ls -A data/corpus 2>/dev/null)" ]; then
    echo -e "\n${BLUE}[1/3] Generating synthetic telecom corpus (500 documents)…${NC}"
    python -m noia.ingestion.corpus_builder --output data/corpus --count 500
else
    echo -e "\n${GREEN}[1/3] Corpus already present — skipping generation.${NC}"
fi

# ── Step 2: Ingest corpus ─────────────────────────────────────────────────────
if [ ! -f data/bm25_index.pkl ]; then
    echo -e "\n${BLUE}[2/3] Ingesting corpus into ChromaDB + BM25 index…${NC}"
    python -m noia.ingestion.ingest_pipeline --corpus-dir data/corpus
else
    echo -e "\n${GREEN}[2/3] Index already present — skipping ingestion.${NC}"
fi

# ── Step 3: Start API ─────────────────────────────────────────────────────────
echo -e "\n${BLUE}[3/3] Starting NOIA API server…${NC}"
echo -e "${GREEN}  API docs:  http://localhost:8000/docs${NC}"
echo -e "${GREEN}  Health:    http://localhost:8000/api/v1/health${NC}"
echo -e "${YELLOW}  Demo creds: noc_engineer_demo / noia_demo_2025${NC}\n"

exec uvicorn noia.api.api_server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info
