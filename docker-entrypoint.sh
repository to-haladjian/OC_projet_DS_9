#!/usr/bin/env bash
# Container entrypoint: optionally build the FAISS index on first start, then run
# the requested service command. The index lives on a mounted volume, so this only
# does real work the first time (or after the volume is cleared).
set -euo pipefail

if [ "${BUILD_INDEX_IF_MISSING:-0}" = "1" ]; then
    if [ ! -f /app/faiss_index/index.faiss ]; then
        if [ ! -f /app/data/events_clean.csv ]; then
            echo "[entrypoint] No cleaned data and no index found — running full pipeline (collect + clean + index)..."
            python -c "from src.pipeline import full_refresh; full_refresh()"
        else
            echo "[entrypoint] Cleaned data found but no FAISS index — building the index..."
            python scripts/build_vector_store.py
        fi
    else
        echo "[entrypoint] FAISS index already present — skipping build."
    fi
fi

# Hand off to the service command (uvicorn API or the Dash UI).
exec "$@"
