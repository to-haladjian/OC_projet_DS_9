"""CLI: clean the raw OpenAgenda export into a structured, RAG-ready dataset.

Thin wrapper around :mod:`src.data.clean`. Reads the raw CSV produced by
``scripts/collect_events.py`` and writes ``data/events_clean.csv`` (the input to FAISS
indexing). Prints a before/after row count and a per-column missing-values report.

Usage:
    python scripts/clean_events.py
    python scripts/clean_events.py --csv data/openagenda_92_events.csv --out data/events_clean.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``src`` importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config
from src.data.clean import clean, null_report, save_clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=config.RAW_CSV,
        help="Path to the raw events CSV (default: <project>/data/openagenda_92_events.csv).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=config.CLEAN_CSV,
        help="Path to write the cleaned CSV (default: <project>/data/events_clean.csv).",
    )
    args = parser.parse_args()

    print(f"Loading raw events from {args.csv} ...")
    raw = pd.read_csv(args.csv)
    print(f"Loaded {len(raw)} raw events.")

    cleaned = clean(raw)
    print(f"Kept {len(cleaned)} events after cleaning "
          f"({len(raw) - len(cleaned)} dropped).")

    print("\nMissing values per column:")
    report = null_report(cleaned)
    for column, count in report.items():
        print(f"  {column:<13} {count:>6} null")

    save_clean(cleaned, args.out)
    print(f"\nSaved cleaned dataset -> {args.out}")


if __name__ == "__main__":
    main()
