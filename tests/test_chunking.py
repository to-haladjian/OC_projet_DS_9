"""Unit tests for Document construction and chunking (no embeddings / no API)."""

from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.indexing.chunking import build_documents, build_event_document, split_documents


@pytest.fixture
def event_row() -> dict:
    return {
        "id": "1",
        "title": "Concert de Jazz",
        "description": "Un super concert au coeur de Nanterre.",
        "summary": "Concert",
        "date_start": "2026-02-15T14:00:00+00:00",
        "date_end": "2026-02-15T15:00:00+00:00",
        "daterange": "Dimanche 15 février, 15h00",
        "venue": "La Seine Musicale",
        "address": "1 île Seguin",
        "city": "Boulogne-Billancourt",
        "postalcode": "92100",
        "department": "92",
        "coordinates": "{'lon': 2.2, 'lat': 48.8}",
        "keywords": "Jazz",
        "conditions": "Tarif unique : 10€",
        "url": "https://openagenda.com/e/1",
        "updatedat": "2026-01-01T00:00:00+00:00",
    }


def test_header_prepends_title_date_and_venue(event_row):
    doc = build_event_document(event_row)
    header = doc.page_content.split("\n\n", 1)[0]
    assert "Titre: Concert de Jazz" in header
    assert "Date: Dimanche 15 février, 15h00" in header
    assert "Lieu: La Seine Musicale, Boulogne-Billancourt (92100)" in header
    # The body follows the header.
    assert doc.page_content.endswith("Un super concert au coeur de Nanterre.")


def test_metadata_limited_to_configured_fields(event_row):
    doc = build_event_document(event_row)
    assert set(doc.metadata).issubset(set(config.METADATA_FIELDS))
    assert doc.metadata["department"] == "92"
    assert doc.metadata["city"] == "Boulogne-Billancourt"
    # Non-metadata columns (e.g. conditions, coordinates) are not leaked into metadata.
    assert "conditions" not in doc.metadata


def test_missing_header_fields_are_skipped(event_row):
    event_row["daterange"] = pd.NA
    event_row["venue"] = pd.NA
    event_row["postalcode"] = pd.NA
    doc = build_event_document(event_row)
    header = doc.page_content.split("\n\n", 1)[0]
    assert "Date:" not in header
    assert "Lieu: Boulogne-Billancourt" in header  # city alone still renders


def test_build_documents_drops_events_without_description(event_row):
    rows = [event_row, {**event_row, "id": "2", "description": pd.NA}]
    docs = build_documents(pd.DataFrame(rows))
    assert len(docs) == 1
    assert docs[0].metadata["id"] == "1"


def test_split_preserves_metadata_and_respects_chunk_size(event_row):
    long_row = {**event_row, "description": "phrase. " * 1000}  # ~8000 chars
    docs = build_documents(pd.DataFrame([long_row]))
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 500 for c in chunks)
    # Metadata (incl. the dedup id) is copied onto every chunk.
    assert all(c.metadata["id"] == "1" for c in chunks)
