"""Unit tests for filter extraction and the metadata predicate (no LLM / no API)."""

from __future__ import annotations

from datetime import date

from src.rag.filters import build_metadata_filter, extract_filters


class _FakeLLM:
    """Stand-in chat model that returns a fixed string as its response content."""

    def __init__(self, reply: str):
        self.reply = reply

    def invoke(self, _messages):
        class _Msg:
            content = self.reply

        return _Msg()


_TODAY = date(2026, 6, 4)


# --- extract_filters: defensive JSON parsing ---

def test_extract_filters_strips_json_code_fence():
    llm = _FakeLLM('```json\n{"city": "Nanterre", "date_from": "2026-06-06"}\n```')
    assert extract_filters("...", llm, _TODAY) == {
        "city": "Nanterre",
        "date_from": "2026-06-06",
    }


def test_extract_filters_recovers_json_from_surrounding_prose():
    llm = _FakeLLM('Bien sûr : {"city": "Antony"} voilà.')
    assert extract_filters("...", llm, _TODAY) == {"city": "Antony"}


def test_extract_filters_drops_null_and_empty_values():
    llm = _FakeLLM('{"city": "Nanterre", "date_from": "", "date_to": null}')
    assert extract_filters("...", llm, _TODAY) == {"city": "Nanterre"}


def test_extract_filters_falls_back_to_empty_dict_on_garbage():
    llm = _FakeLLM("désolé, je n'ai pas compris la question")
    assert extract_filters("...", llm, _TODAY) == {}


def test_extract_filters_ignores_unknown_keys():
    llm = _FakeLLM('{"city": "Paris", "foo": "bar"}')
    assert extract_filters("...", llm, _TODAY) == {"city": "Paris"}


# --- build_metadata_filter: the retrieval predicate ---

def test_no_filters_returns_none():
    assert build_metadata_filter({}) is None


def test_city_predicate_is_case_insensitive():
    predicate = build_metadata_filter({"city": "Nanterre"})
    assert predicate({"city": "nanterre"}) is True
    assert predicate({"city": "Antony"}) is False


def test_date_window_keeps_overlapping_events():
    predicate = build_metadata_filter(
        {"date_from": "2026-06-06", "date_to": "2026-06-07"}
    )
    inside = {
        "date_start": "2026-06-06T20:00:00+00:00",
        "date_end": "2026-06-06T23:00:00+00:00",
    }
    before = {
        "date_start": "2026-06-01T20:00:00+00:00",
        "date_end": "2026-06-01T23:00:00+00:00",
    }
    after = {
        "date_start": "2026-06-10T20:00:00+00:00",
        "date_end": "2026-06-10T23:00:00+00:00",
    }
    assert predicate(inside) is True
    assert predicate(before) is False
    assert predicate(after) is False


def test_date_window_is_lenient_with_unparseable_dates():
    predicate = build_metadata_filter({"date_from": "2026-06-06"})
    # Missing/unparseable event dates -> kept (avoid over-filtering).
    assert predicate({"date_start": "NaT", "date_end": ""}) is True
    assert predicate({}) is True


def test_combined_filters_require_all_conditions():
    predicate = build_metadata_filter(
        {"city": "Nanterre", "date_from": "2026-06-06", "date_to": "2026-06-07"}
    )
    event = {
        "date_start": "2026-06-06T20:00:00+00:00",
        "date_end": "2026-06-06T23:00:00+00:00",
    }
    assert predicate({"city": "Nanterre", **event}) is True
    assert predicate({"city": "Antony", **event}) is False
