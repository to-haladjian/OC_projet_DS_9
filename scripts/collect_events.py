"""CLI: collect Île-de-France events from OpenAgenda for the last year + upcoming.

Thin wrapper around :mod:`src.data.fetch_openagenda`. Writes the raw export to
``data/openagenda_idf_events.csv`` (consumed next by ``scripts/clean_events.py``).

Usage:
    python scripts/collect_events.py
    python scripts/collect_events.py --days 365 --outdir data --all-fields
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

# Make ``src`` importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config
from src.data import fetch_openagenda as fetch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Look-back window in days (default: 365 = last year).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=config.DATA_DIR,
        help="Directory to write the collected events into (default: <project>/data).",
    )
    parser.add_argument(
        "--all-fields",
        action="store_true",
        help="Keep every dataset field instead of the curated subset.",
    )
    args = parser.parse_args()

    since = date.today() - timedelta(days=args.days)
    where = fetch.build_where(since)

    expected = fetch.count_events(where)
    print(f"Collecting Île-de-France events with end date >= {since} ...")
    print(f"Matching events reported by the API: {expected}")

    fields = None if args.all_fields else fetch.DEFAULT_FIELDS
    events = fetch.fetch_events(where, fields=fields)
    print(f"Downloaded {len(events)} events.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outdir / "openagenda_idf_events.csv"

    pd.DataFrame(events).to_csv(csv_path, index=False)

    print(f"Saved CSV  -> {csv_path}")


if __name__ == "__main__":
    main()
