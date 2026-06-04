"""The RAG chain: two Mistral calls around a metadata-pre-filtered FAISS retrieval.

Flow for one question:
  1. **Filter extraction** (Call 1, NER): the question -> structured filters
     (:func:`src.rag.filters.extract_filters`).
  2. **Retrieval**: semantic search over FAISS, pre-filtered on Document metadata; falls
     back to a full-corpus search when the filtered search returns nothing.
  3. **Generation** (Call 2): answer grounded in the retrieved context only.

The logic lives here (separate from any API/CLI) so it can be imported by FastAPI later.
``answer`` is the primary entry point; ``as_runnable`` exposes the same flow as an LCEL
``Runnable`` for composition.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from src import config
from src.indexing.build_index import load_vector_store, make_embeddings
from src.rag.filters import build_metadata_filter, extract_filters
from src.rag.prompts import GENERATION_SYSTEM_PROMPT


def format_docs(docs: list[Document]) -> str:
    """Serialize retrieved documents into the context block fed to the generator."""
    return "\n\n".join(
        f"Source: {doc.metadata}\nContenu: {doc.page_content}" for doc in docs
    )


class RAGChain:
    """Encapsulates the embeddings, FAISS store and the two Mistral models."""

    def __init__(self, index_dir: Path = config.FAISS_DIR, k: int = config.RETRIEVAL_K):
        self.index_dir = Path(index_dir)
        self.k = k
        self.embeddings = make_embeddings()
        self.vector_store = load_vector_store(self.index_dir, self.embeddings)
        self.filter_llm = init_chat_model(
            config.FILTER_MODEL,
            model_provider="mistralai",
            temperature=config.FILTER_TEMPERATURE,
        )
        self.gen_llm = init_chat_model(
            config.GENERATION_MODEL,
            model_provider="mistralai",
            temperature=config.GENERATION_TEMPERATURE,
        )

    def reload(self) -> None:
        """Reload the FAISS store from disk (so a future /rebuild takes effect)."""
        self.vector_store = load_vector_store(self.index_dir, self.embeddings)

    def _retrieve(self, question: str, filters: dict) -> list[Document]:
        """Pre-filtered semantic search, with a full-corpus fallback on 0 hits."""
        predicate = build_metadata_filter(filters)
        if predicate is not None:
            docs = self.vector_store.similarity_search(
                question, k=self.k, filter=predicate
            )
            if docs:
                return docs
        return self.vector_store.similarity_search(question, k=self.k)

    def _generate(self, question: str, context: str, today: date) -> str:
        system = GENERATION_SYSTEM_PROMPT.format(today=today.isoformat())
        human = f"Contexte :\n{context}\n\nQuestion : {question}"
        response = self.gen_llm.invoke([("system", system), ("human", human)])
        return getattr(response, "content", str(response))

    def answer(self, question: str) -> dict:
        """Answer ``question`` and return the answer, the extracted filters and sources."""
        today = date.today()
        filters = extract_filters(question, self.filter_llm, today)
        docs = self._retrieve(question, filters)
        context = format_docs(docs)
        answer = self._generate(question, context, today)
        return {
            "answer": answer,
            "filters": filters,
            "sources": [doc.metadata for doc in docs],
        }

    def as_runnable(self) -> Runnable:
        """Expose the same flow as an LCEL Runnable taking ``{"question": ...}``."""

        def _filters(inp: dict) -> dict:
            return extract_filters(inp["question"], self.filter_llm, date.today())

        def _retrieve(inp: dict) -> list[Document]:
            return self._retrieve(inp["question"], inp["filters"])

        def _generate(inp: dict) -> str:
            return self._generate(
                inp["question"], format_docs(inp["context"]), date.today()
            )

        return (
            RunnablePassthrough.assign(filters=RunnableLambda(_filters))
            | RunnablePassthrough.assign(context=RunnableLambda(_retrieve))
            | RunnableLambda(_generate)
        )
