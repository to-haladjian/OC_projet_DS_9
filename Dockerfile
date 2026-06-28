# Multi-stage build: one image shared by both services (API + Dash UI).
# Stage 1 resolves the Poetry dependencies into a self-contained virtualenv;
# stage 2 copies only that venv plus the application code, keeping the image small.

# --- Stage 1: builder ---------------------------------------------------------
FROM python:3.13-slim AS builder

ENV POETRY_VERSION=2.3.2 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install "poetry==${POETRY_VERSION}"

WORKDIR /app

# Copy only the dependency manifests first so this layer is cached unless they change.
COPY pyproject.toml poetry.lock ./

# Install runtime dependencies only (no dev group, no project package — package-mode=false).
RUN poetry install --only main --no-root

# --- Stage 2: runtime ---------------------------------------------------------
FROM python:3.13-slim AS runtime

# curl is used by the compose healthcheck against /health.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Bring in the resolved virtualenv from the builder stage.
COPY --from=builder /app/.venv /app/.venv

# Application code (data/ and faiss_index/ are provided at runtime via volumes).
COPY src/ ./src/
COPY interface/ ./interface/
COPY scripts/ ./scripts/
COPY docker-entrypoint.sh ./docker-entrypoint.sh
RUN chmod +x ./docker-entrypoint.sh

EXPOSE 8000 8050

# The entrypoint builds the index on first start when BUILD_INDEX_IF_MISSING=1,
# then execs the service command (default: the API).
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
