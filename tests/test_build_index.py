"""Unit tests for the vectorisation/indexing pipeline (no Mistral calls).

A tiny deterministic fake stands in for ``mistral-embed`` so ``FAISS.from_documents`` runs
fully in-memory: we test the batching loop, the empty-input guard, the string typing of
location codes, and a build -> load -> search round-trip.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.indexing import build_index as bi


class _FakeEmbeddings(Embeddings):
    """Deterministic stand-in for ``MistralAIEmbeddings`` (stable 8-dim vectors)."""

    dim = 8

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i] / 255.0 for i in range(self.dim)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def _docs(n: int) -> list[Document]:
    return [
        Document(page_content=f"Événement numéro {i}", metadata={"id": str(i)})
        for i in range(n)
    ]


def test_embed_and_index_rejects_empty_input():
    with pytest.raises(ValueError):
        bi.embed_and_index([], _FakeEmbeddings(), batch_size=2)


def test_embed_and_index_indexes_every_document_across_batches():
    documents = _docs(5)
    store = bi.embed_and_index(documents, _FakeEmbeddings(), batch_size=2)

    indexed_ids = {doc.metadata["id"] for doc in store.docstore._dict.values()}
    assert indexed_ids == {"0", "1", "2", "3", "4"}


def test_load_events_keeps_location_codes_as_strings(tmp_path):
    csv_path = tmp_path / "events_clean.csv"
    pd.DataFrame(
        [{"id": "1", "postalcode": "92000", "department": "92", "description": "x"}]
    ).to_csv(csv_path, index=False)

    df = bi.load_events(csv_path)
    assert df["postalcode"].dtype == "string"
    assert df["department"].dtype == "string"
    assert df.loc[0, "postalcode"] == "92000"


def test_build_index_round_trips_through_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(bi, "make_embeddings", lambda: _FakeEmbeddings())
    csv_path = tmp_path / "events_clean.csv"
    pd.DataFrame(
        [
            {"id": "1", "title": "Concert", "description": "Un concert de jazz à Nanterre."},
            {"id": "2", "title": "Expo", "description": "Une exposition d'art moderne."},
        ]
    ).to_csv(csv_path, index=False)

    outdir = tmp_path / "faiss_index"
    bi.build_index(csv_path=csv_path, outdir=outdir, no_chunk=True)
    assert (outdir / "index.faiss").exists()

    store = bi.load_vector_store(outdir, _FakeEmbeddings())
    hits = store.similarity_search("jazz", k=1)
    assert len(hits) == 1
    assert str(hits[0].metadata["id"]) in {"1", "2"}
