"""End-to-end data pipeline orchestration, used by the API's ``/rebuild`` endpoint.

Composes the existing building blocks — collection, cleaning and indexing — into a
single ``full_refresh`` that re-fetches OpenAgenda events, re-cleans them and rebuilds the
FAISS index. The index is built into a sibling temp directory and then **atomically
swapped** into place, so a concurrent reader never sees a half-written index.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from src import config
from src.data.clean import clean, save_clean
from src.data.fetch_openagenda import collect_events
from src.indexing.build_index import build_index


def refresh_data(days: int = 365, csv_path: Path = config.CLEAN_CSV) -> Path:
    """Re-collect events from OpenAgenda, clean them and write ``csv_path``."""
    raw = collect_events(days=days)
    cleaned = clean(raw)
    save_clean(cleaned, csv_path)
    return csv_path


def rebuild_index_atomic(
    csv_path: Path = config.CLEAN_CSV,
    outdir: Path = config.FAISS_DIR,
    **build_kwargs,
) -> Path:
    """Build the FAISS index into a temp dir, then atomically replace ``outdir``."""
    outdir = Path(outdir)
    new_dir = outdir.with_name(outdir.name + ".new")
    old_dir = outdir.with_name(outdir.name + ".old")

    # Build into a fresh temp directory (never touches the live index).
    shutil.rmtree(new_dir, ignore_errors=True)
    build_index(csv_path=csv_path, outdir=new_dir, **build_kwargs)

    # Swap: move the live index aside, promote the new one, drop the old one.
    shutil.rmtree(old_dir, ignore_errors=True)
    if outdir.exists():
        outdir.rename(old_dir)
    new_dir.rename(outdir)
    shutil.rmtree(old_dir, ignore_errors=True)
    return outdir


def full_refresh(days: int = 365) -> Path:
    """Re-collect + re-clean the data, then atomically rebuild the FAISS index."""
    csv_path = refresh_data(days=days)
    return rebuild_index_atomic(csv_path=csv_path)
