# ─────────────────────────────────────────────────────────────────────────────
# NOIA — Network Operations Intelligence Assistant
# Multi-stage Dockerfile
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Python dependencies ─────────────────────────────────────────────
FROM python:3.11-slim AS python-deps

WORKDIR /install

# System deps for native Python extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install/pkg -r requirements.txt


# ── Stage 2: Runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="AITR-NOIA Team <kwame.asante@aitr.org.gh>"
LABEL org.opencontainers.image.title="NOIA NOC Intelligence Assistant"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# Runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from build stage
COPY --from=python-deps /install/pkg /usr/local

WORKDIR /app

# Copy application source
COPY noia/       ./noia/
COPY config/     ./config/
COPY evaluation/ ./evaluation/
COPY scripts/    ./scripts/
COPY .env.example .env.example

# Create runtime directories
RUN mkdir -p data/chroma data/corpus models results

# Non-root user for security
RUN useradd -m -u 1001 noia && chown -R noia:noia /app
USER noia

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "noia.api.api_server:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "info"]
