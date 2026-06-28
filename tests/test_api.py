"""Deterministic API tests using a fake RAG chain (no FAISS load, no Mistral calls).

The real ``RAGChain`` is heavy (loads the index + LLMs at startup), so we never let the
app's lifespan build one: we override the ``get_rag_chain`` dependency and inject the
app state the routes need, then drive the app with FastAPI's ``TestClient``.
"""

from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from src.api import main
from src.api.main import app, get_rag_chain


class _FakeChain:
    """Canned stand-in for RAGChain used across the API tests."""

    def __init__(self):
        self.reloaded = False

        class _Store:
            docstore = type("D", (), {"_dict": {"a": 1, "b": 2}})()

        self.vector_store = _Store()

    def answer(self, question: str) -> dict:
        return {
            "answer": f"Réponse à : {question}",
            "filters": {"city": "Nanterre"},
            "sources": [{"title": "Concert", "city": "Nanterre"}],
        }

    def reload(self) -> None:
        self.reloaded = True


@pytest.fixture
def client():
    """TestClient with the RAG chain faked and app state injected manually.

    We deliberately do NOT use ``TestClient`` as a context manager: that would run the
    app lifespan, which builds a real ``RAGChain`` (loading FAISS + Mistral) and would
    overwrite our fake. Skipping lifespan keeps these tests offline and deterministic.
    """
    fake = _FakeChain()
    app.state.rag_chain = fake
    app.state.rebuild_lock = threading.Lock()
    app.dependency_overrides[get_rag_chain] = lambda: fake
    test_client = TestClient(app)
    test_client.fake = fake
    yield test_client
    app.dependency_overrides.clear()


# --- /health ---

def test_health_reports_document_count(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "documents": 2}


# --- /metadata ---

def test_metadata_aggregates_corpus_stats(client):
    # Two chunks of event "1" (must be deduplicated) plus event "2".
    docs = {
        "k1": Document(page_content="...", metadata={
            "id": "1", "city": "Nanterre", "department": "92",
            "date_start": "2026-06-01T20:00:00+00:00",
            "date_end": "2026-06-01T23:00:00+00:00"}),
        "k2": Document(page_content="...", metadata={
            "id": "1", "city": "Nanterre", "department": "92",
            "date_start": "2026-06-01T20:00:00+00:00",
            "date_end": "2026-06-01T23:00:00+00:00"}),
        "k3": Document(page_content="...", metadata={
            "id": "2", "city": "Boulogne-Billancourt", "department": "92",
            "date_start": "2026-07-15T10:00:00+00:00",
            "date_end": "2026-07-16T18:00:00+00:00"}),
    }
    client.fake.vector_store = type(
        "S", (), {"docstore": type("D", (), {"_dict": docs})()}
    )()

    resp = client.get("/metadata")
    assert resp.status_code == 200
    body = resp.json()
    assert body["events"] == 2  # deduplicated by id
    assert body["cities"] == 2
    assert body["departments"] == {"92": 2}
    assert body["date_range"] == {"from": "2026-06-01", "to": "2026-07-16"}


# --- /ask ---

def test_ask_returns_answer_filters_and_sources(client):
    resp = client.post("/ask", json={"question": "Concerts à Nanterre ?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Réponse à : Concerts à Nanterre ?"
    assert body["filters"] == {"city": "Nanterre"}
    assert body["sources"] == [{"title": "Concert", "city": "Nanterre"}]


def test_ask_rejects_blank_question(client):
    assert client.post("/ask", json={"question": "   "}).status_code == 422


def test_ask_rejects_missing_field(client):
    assert client.post("/ask", json={}).status_code == 422


def test_ask_returns_502_on_backend_failure(client):
    def boom(_question):
        raise RuntimeError("mistral down")

    client.fake.answer = boom
    assert client.post("/ask", json={"question": "x"}).status_code == 502


# --- /rebuild ---

def test_rebuild_schedules_and_reloads(client, monkeypatch):
    calls = {"refresh": 0}
    monkeypatch.setattr(main, "full_refresh", lambda *a, **k: calls.__setitem__("refresh", calls["refresh"] + 1))

    resp = client.post("/rebuild")
    assert resp.status_code == 202
    assert resp.json() == {"status": "rebuild scheduled"}
    # Background task ran on client context exit: refresh happened and chain reloaded.
    assert calls["refresh"] == 1
    assert client.fake.reloaded is True


def test_rebuild_conflicts_when_already_running(client):
    # Simulate an in-progress rebuild by holding the lock.
    app.state.rebuild_lock.acquire()
    try:
        assert client.post("/rebuild").status_code == 409
    finally:
        app.state.rebuild_lock.release()


def test_rebuild_requires_token_when_configured(client, monkeypatch):
    monkeypatch.setenv("REBUILD_TOKEN", "secret")
    monkeypatch.setattr(main, "full_refresh", lambda *a, **k: None)

    assert client.post("/rebuild").status_code == 401
    assert client.post("/rebuild", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/rebuild", headers={"X-API-Key": "secret"}).status_code == 202
