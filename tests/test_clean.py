"""Unit tests for the data cleaning pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import SCHEMA_COLUMNS
from src.data import clean as clean_mod
from src.data.clean import (
    clean,
    normalize_keywords,
    normalize_postalcode,
    resolve_department,
    strip_html,
)


# --- Field-level helpers ---

def test_strip_html_removes_tags_and_unescapes_entities():
    assert strip_html("<p>Un <b>super</b> concert &agrave; Paris</p>") == (
        "Un super concert à Paris"
    )


def test_strip_html_missing_returns_empty_string():
    assert strip_html(None) == ""
    assert strip_html(pd.NA) == ""


def test_normalize_postalcode_float_becomes_five_char_string():
    assert normalize_postalcode(75014.0) == "75014"
    assert normalize_postalcode(7500.0) == "07500"  # zero-padded
    assert pd.isna(normalize_postalcode(None))


def test_resolve_department_prefers_postalcode_then_name():
    assert resolve_department("75014", "anything") == "75"
    assert resolve_department(pd.NA, "Val-D'Oise") == "95"
    assert resolve_department(pd.NA, "Seine-St-Denis") == "93"
    assert pd.isna(resolve_department(pd.NA, "Métropole de Lyon"))


def test_normalize_keywords_flattens_stringified_list():
    assert normalize_keywords("['Jazz', 'Concert']") == "Jazz, Concert"
    assert pd.isna(normalize_keywords(None))


# --- End-to-end cleaning over the fixture ---

@pytest.fixture
def cleaned(raw_events) -> pd.DataFrame:
    return clean(raw_events)


def test_output_has_exactly_the_schema_columns(cleaned):
    assert list(cleaned.columns) == SCHEMA_COLUMNS


def test_html_is_stripped_in_description(cleaned):
    descriptions = cleaned["description"].tolist()
    assert all("<" not in d for d in descriptions)
    assert "Un super concert à Paris" in descriptions


def test_dates_are_datetime_with_nat_for_missing(cleaned):
    assert pd.api.types.is_datetime64_any_dtype(cleaned["date_start"])
    expo = cleaned.loc[cleaned["title"] == "Expo"].iloc[0]
    assert pd.isna(expo["date_start"])  # row 3 had no start date


def test_postalcode_and_department_are_normalized(cleaned):
    paris = cleaned.loc[cleaned["id"] == "1"].iloc[0]
    assert paris["postalcode"] == "75014"
    assert paris["department"] == "75"


def test_duplicate_id_collapses_to_newest(cleaned):
    rows = cleaned.loc[cleaned["id"] == "1"]
    assert len(rows) == 1
    assert rows.iloc[0]["title"] == "Concert (mis à jour)"


def test_non_idf_event_is_dropped(cleaned):
    assert "4" not in set(cleaned["id"])


def test_textless_event_is_dropped(cleaned):
    assert "5" not in set(cleaned["id"])


def test_department_recovered_from_name_when_no_postalcode(cleaned):
    festival = cleaned.loc[cleaned["id"] == "6"].iloc[0]
    assert festival["department"] == "95"
    assert pd.isna(festival["postalcode"])


def test_save_clean_writes_csv(tmp_path, cleaned):
    out = tmp_path / "nested" / "events_clean.csv"
    clean_mod.save_clean(cleaned, out)
    assert out.exists()
    reloaded = pd.read_csv(out)
    assert list(reloaded.columns) == SCHEMA_COLUMNS
    assert len(reloaded) == len(cleaned)
