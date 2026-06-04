"""Unit tests for the OpenAgenda acquisition logic (no network calls)."""

from __future__ import annotations

from datetime import date

import pytest

from src.data import fetch_openagenda as fetch


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_build_where_formats_date_filter():
    assert fetch.build_where(date(2024, 1, 1)) == "lastdate_end >= date'2024-01-01'"


def test_default_fields_cover_key_columns():
    # The text we embed and the dedup key must be requested from the API.
    for column in ("uid", "title_fr", "longdescription_fr", "lastdate_end"):
        assert column in fetch.DEFAULT_FIELDS


def test_fetch_events_uses_export_endpoint_with_select(monkeypatch):
    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return _FakeResponse([{"uid": "1"}])

    monkeypatch.setattr(fetch.requests, "get", fake_get)

    result = fetch.fetch_events("clause", fields=["uid", "title_fr"])

    assert captured["url"] == fetch.EXPORT_URL
    assert captured["params"]["where"] == "clause"
    assert captured["params"]["select"] == "uid,title_fr"
    assert result == [{"uid": "1"}]


def test_fetch_events_without_fields_omits_select(monkeypatch):
    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured["params"] = params
        return _FakeResponse([])

    monkeypatch.setattr(fetch.requests, "get", fake_get)

    fetch.fetch_events("clause", fields=None)

    assert "select" not in captured["params"]


def test_count_events_returns_total_count(monkeypatch):
    monkeypatch.setattr(
        fetch.requests, "get", lambda *a, **k: _FakeResponse({"total_count": 42})
    )
    assert fetch.count_events("clause") == 42
