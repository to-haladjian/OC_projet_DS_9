"""Unit tests for the RAG chain's retrieval + generation logic (no FAISS, no Mistral).

The real ``RAGChain.__init__`` loads the FAISS index and the Mistral models, so we build
the object with ``__new__`` and inject fakes for the vector store and the generation LLM.
This isolates ``_retrieve`` (incl. the full-corpus fallback), ``_generate`` and ``answer``.
"""

from __future__ import annotations

from datetime import date

from langchain_core.documents import Document

from src.rag import chain as chain_mod
from src.rag.chain import RAGChain, format_docs


class _FakeStore:
    """Records its similarity_search calls and returns canned, filter-dependent hits."""

    def __init__(self, filtered: list[Document], unfiltered: list[Document]):
        self.filtered = filtered
        self.unfiltered = unfiltered
        self.calls: list[dict] = []

    def similarity_search(self, query, k, filter=None):
        self.calls.append({"query": query, "k": k, "filter": filter})
        return self.filtered if filter is not None else self.unfiltered


class _FakeLLM:
    """Returns a canned message; records the messages it was invoked with."""

    def __init__(self, content: str):
        self.content = content
        self.invoked_with = None

    def invoke(self, messages):
        self.invoked_with = messages
        return type("Msg", (), {"content": self.content})()


def _chain(store=None, gen_llm=None, k=6) -> RAGChain:
    chain = RAGChain.__new__(RAGChain)
    chain.k = k
    chain.vector_store = store
    chain.gen_llm = gen_llm
    chain.filter_llm = None  # unused: extract_filters is faked where needed
    return chain


# --- format_docs ---

def test_format_docs_serializes_source_and_content():
    docs = [Document(page_content="Concert", metadata={"id": "1", "city": "Paris"})]
    block = format_docs(docs)
    assert "Source:" in block
    assert "Contenu: Concert" in block
    assert "Paris" in block


# --- _retrieve ---

def test_retrieve_returns_filtered_hits_when_predicate_matches():
    filtered = [Document(page_content="A", metadata={"id": "a"})]
    store = _FakeStore(filtered=filtered, unfiltered=[Document(page_content="B", metadata={"id": "b"})])
    chain = _chain(store=store)

    docs = chain._retrieve("question", {"department": "75"})
    assert [d.metadata["id"] for d in docs] == ["a"]
    assert len(store.calls) == 1
    assert store.calls[0]["filter"] is not None  # pre-filter applied


def test_retrieve_falls_back_to_full_corpus_on_zero_filtered_hits():
    fallback = [Document(page_content="B", metadata={"id": "b"})]
    store = _FakeStore(filtered=[], unfiltered=fallback)
    chain = _chain(store=store)

    docs = chain._retrieve("question", {"department": "75"})
    assert [d.metadata["id"] for d in docs] == ["b"]
    # Two calls: pre-filtered (empty) then full-corpus fallback (filter=None).
    assert len(store.calls) == 2
    assert store.calls[0]["filter"] is not None
    assert store.calls[1]["filter"] is None


def test_retrieve_skips_prefilter_when_no_filters():
    store = _FakeStore(filtered=[], unfiltered=[Document(page_content="B", metadata={"id": "b"})])
    chain = _chain(store=store)

    docs = chain._retrieve("question", {})
    assert [d.metadata["id"] for d in docs] == ["b"]
    assert len(store.calls) == 1
    assert store.calls[0]["filter"] is None


# --- _generate ---

def test_generate_invokes_llm_and_returns_content():
    llm = _FakeLLM("Voici les événements.")
    chain = _chain(gen_llm=llm)

    out = chain._generate("Quoi ?", "contexte", date(2026, 6, 28))
    assert out == "Voici les événements."
    assert llm.invoked_with is not None  # the LLM was actually called


# --- answer ---

def test_answer_returns_answer_filters_and_sources(monkeypatch):
    monkeypatch.setattr(chain_mod, "extract_filters", lambda q, llm, today: {"city": "Paris"})
    docs = [Document(page_content="Concert", metadata={"id": "1", "title": "Jazz"})]
    store = _FakeStore(filtered=docs, unfiltered=docs)
    chain = _chain(store=store, gen_llm=_FakeLLM("Réponse ancrée."))

    result = chain.answer("Concerts à Paris ?")
    assert result["answer"] == "Réponse ancrée."
    assert result["filters"] == {"city": "Paris"}
    assert result["sources"] == [{"id": "1", "title": "Jazz"}]
    assert "contexts" not in result


def test_answer_includes_contexts_when_requested(monkeypatch):
    monkeypatch.setattr(chain_mod, "extract_filters", lambda q, llm, today: {})
    docs = [Document(page_content="Contenu du concert", metadata={"id": "1"})]
    store = _FakeStore(filtered=docs, unfiltered=docs)
    chain = _chain(store=store, gen_llm=_FakeLLM("ok"))

    result = chain.answer("Concerts ?", return_contexts=True)
    assert result["contexts"] == ["Contenu du concert"]
