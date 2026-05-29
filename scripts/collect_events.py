"""Collect Île-de-France events from OpenAgenda for the last year and upcoming events.

Source: the official "Événements publics en Île-de-France (via Open Agenda)" open dataset
(`evenements-publics-cibul`), exposed through the Opendatasoft Explore API v2.1.
It aggregates OpenAgenda public events for the Île-de-France region — no API key required.

API reference: https://data.iledefrance.fr/explore/dataset/evenements-publics-cibul/api/
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

# Project root (the directory above this script's ``scripts/`` folder), so default
# data paths resolve there regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_URL = (
    "https://data.iledefrance.fr/api/explore/v2.1/catalog/datasets"
    "/evenements-publics-cibul"
)
RECORDS_URL = f"{BASE_URL}/records"
EXPORT_URL = f"{BASE_URL}/exports/json"

# Curated, RAG-friendly subset of the dataset's fields. Set to None to keep every field.
DEFAULT_FIELDS = [
    "uid",
    "slug",
    "canonicalurl",
    "title_fr",
    "description_fr",
    "longdescription_fr",
    "conditions_fr",
    "keywords_fr",
    "daterange_fr",
    "firstdate_begin",
    "firstdate_end",
    "lastdate_begin",
    "lastdate_end",
    "location_name",
    "location_address",
    "location_postalcode",
    "location_city",
    "location_department",
    "location_region",
    "location_coordinates",
    "age_min",
    "age_max",
    "registration",
    "originagenda_title",
    "updatedat",
]


def count_events(where: str, timeout: int = 30) -> int:
    """Return how many events match the ``where`` clause (cheap, limit=0)."""
    resp = requests.get(
        RECORDS_URL, params={"where": where, "limit": 0}, timeout=timeout
    )
    resp.raise_for_status()
    return resp.json()["total_count"]


def fetch_events(
    where: str, fields: list[str] | None = DEFAULT_FIELDS, timeout: int = 300
) -> list[dict]:
    """Download every event matching ``where`` via the bulk export endpoint.

    The export endpoint streams the full filtered result set in one request, so it
    sidesteps the 10,000-record offset limit of the paginated /records endpoint.
    """
    params: dict[str, str] = {"where": where}
    if fields:
        params["select"] = ",".join(fields)
    resp = requests.get(EXPORT_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def build_where(since: date) -> str:
    """Match events that overlap [since, +inf): last-year, ongoing and upcoming.

    An event is kept when its end date is on or after ``since``; this captures events
    that took place during the past year as well as everything scheduled in the future.
    """
    return f"lastdate_end >= date'{since.isoformat()}'"


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
        default=PROJECT_ROOT / "data",
        help="Directory to write the collected events into (default: <project>/data).",
    )
    parser.add_argument(
        "--all-fields",
        action="store_true",
        help="Keep every dataset field instead of the curated subset.",
    )
    args = parser.parse_args()

    since = date.today() - timedelta(days=args.days)
    where = build_where(since)

    expected = count_events(where)
    print(f"Collecting Île-de-France events with end date >= {since} ...")
    print(f"Matching events reported by the API: {expected}")

    events = fetch_events(where, fields=None if args.all_fields else DEFAULT_FIELDS)
    print(f"Downloaded {len(events)} events.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    json_path = args.outdir / "openagenda_idf_events.json"
    csv_path = args.outdir / "openagenda_idf_events.csv"

    json_path.write_text(
        json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(events).to_csv(csv_path, index=False)

    print(f"Saved JSON -> {json_path}")
    print(f"Saved CSV  -> {csv_path}")


if __name__ == "__main__":
    main()
