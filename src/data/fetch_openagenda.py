"""Fetch Île-de-France events from OpenAgenda (Opendatasoft Explore API v2.1).

Source: the official "Événements publics en Île-de-France (via Open Agenda)" open
dataset (``evenements-publics-cibul``). It aggregates OpenAgenda public events for the
Île-de-France region — no API key required.

API reference: https://data.iledefrance.fr/explore/dataset/evenements-publics-cibul/api/

This module holds the reusable acquisition logic; ``scripts/collect_events.py`` is the
thin CLI wrapper around it.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import requests

BASE_URL = (
    "https://data.iledefrance.fr/api/explore/v2.1/catalog/datasets"
    "/evenements-publics-cibul"
)
RECORDS_URL = f"{BASE_URL}/records"
EXPORT_URL = f"{BASE_URL}/exports/json"

# Curated, RAG-friendly subset of the dataset's fields. Pass ``all_fields=True`` to keep
# every field instead.
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


def collect_events(days: int = 365, all_fields: bool = False) -> pd.DataFrame:
    """Fetch the events with an end date within the last ``days`` (or upcoming).

    Convenience wrapper returning a DataFrame, used by the CLI and by tests.
    """
    since = date.today() - timedelta(days=days)
    where = build_where(since)
    events = fetch_events(where, fields=None if all_fields else DEFAULT_FIELDS)
    return pd.DataFrame(events)
