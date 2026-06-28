"""Shared fixtures for the data-pipeline tests.

``raw_events`` is a tiny in-memory stand-in for the OpenAgenda export, crafted to exercise
every cleaning edge case: HTML descriptions, a duplicate id, a missing-date event, an
out-of-scope (non-92) event, a text-less event, and an event whose Hauts-de-Seine
department must be recovered from its free-text name (no postal code).
"""

from __future__ import annotations

import pandas as pd
import pytest

# Every column ``src.data.clean.clean`` reads, with sensible defaults; individual rows
# override only what matters for the case they test.
_RAW_COLUMNS = {
    "uid": "0",
    "slug": "slug",
    "canonicalurl": "https://openagenda.com/e/0",
    "title_fr": "Titre",
    "description_fr": "Court résumé",
    "longdescription_fr": "<p>Description</p>",
    "conditions_fr": "Entrée gratuite",
    "keywords_fr": "['Jazz']",
    "daterange_fr": "Dimanche 15 février, 15h00",
    "firstdate_begin": "2026-02-15T14:00:00+00:00",
    "firstdate_end": "2026-02-15T15:00:00+00:00",
    "lastdate_begin": "2026-02-15T14:00:00+00:00",
    "lastdate_end": "2026-02-15T15:00:00+00:00",
    "location_name": "Salle",
    "location_address": "1 rue X",
    "location_postalcode": 92000.0,
    "location_city": "Nanterre",
    "location_department": "Hauts-de-Seine",
    "location_region": "Île-de-France",
    "location_coordinates": "{'lon': 2.3, 'lat': 48.8}",
    "age_min": None,
    "age_max": None,
    "registration": None,
    "originagenda_title": "Agenda",
    "updatedat": "2026-01-01T00:00:00+00:00",
}


def _row(**overrides) -> dict:
    return {**_RAW_COLUMNS, **overrides}


@pytest.fixture
def raw_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # 1. Normal Hauts-de-Seine event, HTML body + entity, oldest update of id "1".
            _row(
                uid="1",
                title_fr="Concert &amp; Jazz",
                longdescription_fr="<p>Un <b>super</b> concert &agrave; Nanterre</p>",
                location_postalcode=92000.0,
                location_department="Hauts-de-Seine",
                updatedat="2026-01-01T00:00:00+00:00",
            ),
            # 2. Duplicate id "1" but newer -> dedup keeps this row (title + body).
            _row(
                uid="1",
                title_fr="Concert (mis à jour)",
                longdescription_fr="<p>Un <b>super</b> concert &agrave; Nanterre</p>",
                location_postalcode=92000.0,
                updatedat="2026-03-01T00:00:00+00:00",
            ),
            # 3. Missing start date, valid Hauts-de-Seine event -> kept, date_start NaT.
            _row(
                uid="3",
                title_fr="Expo",
                firstdate_begin=None,
                location_postalcode=92100.0,
                location_city="Boulogne-Billancourt",
                location_department="Hauts-de-Seine",
            ),
            # 4. Out-of-scope (Lyon, non-92) -> dropped.
            _row(
                uid="4",
                title_fr="Évènement Lyon",
                location_postalcode=69001.0,
                location_department="Métropole de Lyon",
                location_city="Lyon",
            ),
            # 5. No usable text at all -> dropped.
            _row(
                uid="5",
                title_fr=None,
                description_fr=None,
                longdescription_fr=None,
                location_postalcode=92001.0,
            ),
            # 6. No postal code, department recovered from name variant -> kept as 92.
            _row(
                uid="6",
                title_fr="Festival",
                location_postalcode=None,
                location_city="Antony",
                location_department="Hauts-de-Seine",
            ),
        ]
    )
