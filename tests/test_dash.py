"""Unit tests for the Dash interface helpers (no server, no network)."""

from __future__ import annotations

import requests

from interface import dash_app
from interface.dash_app import ask_api, render_message


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _texts(component) -> str:
    """Flatten a Dash component tree into its concatenated text (for assertions)."""
    if isinstance(component, str):
        return component
    if isinstance(component, list):
        return " ".join(_texts(c) for c in component)
    children = getattr(component, "children", None)
    text = _texts(children) if children is not None else ""
    href = getattr(component, "href", "") or ""
    return f"{text} {href}"


# --- ask_api ---

def test_ask_api_returns_json_on_success(monkeypatch):
    payload = {"answer": "ok", "filters": {}, "sources": []}
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(payload))
    assert ask_api("Une question ?", "http://x") == payload


def test_ask_api_returns_error_on_network_failure(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", boom)
    result = ask_api("Une question ?", "http://x")
    assert "error" in result and "refused" in result["error"]


# --- render_message ---

def test_render_user_message_shows_question():
    comp = render_message({"role": "user", "content": "Concerts à Paris ?"})
    assert "Concerts à Paris ?" in _texts(comp)


def test_render_assistant_message_shows_answer_filters_and_sources():
    entry = {
        "role": "assistant",
        "content": "Un concert a lieu à Paris.",
        "filters": {"city": "Paris"},
        "sources": [
            {"title": "Concert Fishers", "venue": "Le Gymnase", "city": "Paris",
             "daterange": "21 juin", "url": "https://example.com/e/1"}
        ],
    }
    text = _texts(render_message(entry))
    assert "Un concert a lieu à Paris." in text
    assert "Concert Fishers" in text
    assert "https://example.com/e/1" in text
    assert "city = Paris" in text


def test_render_error_message():
    text = _texts(render_message({"role": "assistant", "error": "API injoignable : boom"}))
    assert "API injoignable : boom" in text


def test_render_dedupes_repeated_sources():
    entry = {
        "role": "assistant",
        "content": "x",
        "sources": [
            {"title": "A", "url": "u1"},
            {"title": "A", "url": "u1"},
            {"title": "B", "url": "u2"},
        ],
    }
    # Two unique URLs -> the collapsible summary reports 2 sources.
    assert "Sources (2)" in _texts(render_message(entry))


def test_app_layout_has_core_components():
    # Smoke check that the module builds a Dash app with the expected control ids.
    ids = {c for c in dir(dash_app)}
    assert "app" in ids
    layout_text = str(dash_app.app.layout)
    for component_id in ("history", "chat-window", "question", "send", "reset"):
        assert component_id in layout_text
