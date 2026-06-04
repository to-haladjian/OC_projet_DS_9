"""CLI: build the FAISS vector store from the cleaned OpenAgenda events.

Thin wrapper around :mod:`src.indexing.build_index`. Reads ``data/events_clean.csv``
(produced by ``clean_events.py``), embeds the events with Mistral and persists a FAISS
index to ``faiss_index/`` so the RAG retriever can load it without re-embedding.

Usage:
    python scripts/build_vector_store.py
    python scripts/build_vector_store.py --limit 200          # quick smoke run
    python scripts/build_vector_store.py --csv data/events_clean.csv --outdir faiss_index
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``src`` importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.indexing.build_index import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=config.CLEAN_CSV,
        help="Path to the cleaned events CSV (default: <project>/data/events_clean.csv).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=config.FAISS_DIR,
        help="Directory to save the FAISS index into (default: <project>/faiss_index).",
    )
    parser.add_argument(
        "--no-chunk",
        action="store_true",
        help="Index each event as a single document instead of splitting it into chunks.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=config.CHUNK_SIZE,
        help=f"Max characters per chunk (default: {config.CHUNK_SIZE}). Ignored with --no-chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=config.CHUNK_OVERLAP,
        help=f"Character overlap between chunks (default: {config.CHUNK_OVERLAP}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Chunks per embedding API call (default: 64).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only index the first N events (handy for a quick test run).",
    )
    args = parser.parse_args()

    build_index(
        csv_path=args.csv,
        outdir=args.outdir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        limit=args.limit,
        no_chunk=args.no_chunk,
    )


if __name__ == "__main__":
    main()
