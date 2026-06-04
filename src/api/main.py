"""FastAPI application exposing the RAG chain.

Routing only — all business logic lives in :mod:`src.rag` / :mod:`src.pipeline`. A single
:class:`~src.rag.chain.RAGChain` is built at startup and shared across requests; ``/ask``
reads it fresh on every call, and ``/rebuild`` swaps the underlying index then calls
``reload()``, so answers always reflect the latest index.

Run with::

    poetry run uvicorn src.api.main:app        # Swagger UI at /docs
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request

from src.api.schemas import AskRequest, AskResponse, HealthResponse, RebuildResponse
from src.pipeline import full_refresh
from src.rag.chain import RAGChain


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RAG chain (FAISS + Mistral models) once, on startup."""
    app.state.rag_chain = RAGChain()
    app.state.rebuild_lock = threading.Lock()
    yield


app = FastAPI(
    title="RAG Événements Culturels Île-de-France",
    description="API de questions-réponses sur les événements culturels d'Île-de-France.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Dependencies ---

def get_rag_chain(request: Request) -> RAGChain:
    """Return the shared RAG chain (overridable in tests via dependency_overrides)."""
    return request.app.state.rag_chain


def require_rebuild_token(x_api_key: str | None = Header(default=None)) -> None:
    """Guard ``/rebuild`` with a shared secret; a no-op when REBUILD_TOKEN is unset."""
    expected = os.environ.get("REBUILD_TOKEN")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# --- Routes ---

@app.get("/health", response_model=HealthResponse, summary="Liveness + index size")
def health(chain: RAGChain = Depends(get_rag_chain)) -> HealthResponse:
    """Report that the service is up and how many documents are indexed."""
    documents = len(chain.vector_store.docstore._dict)
    return HealthResponse(status="ok", documents=documents)


@app.post("/ask", response_model=AskResponse, summary="Answer a question")
def ask(
    payload: AskRequest, chain: RAGChain = Depends(get_rag_chain)
) -> AskResponse:
    """Answer a natural-language question, grounded in the retrieved events.

    Returns the generated answer, the structured filters extracted from the question and
    the metadata of the events used as sources.
    """
    try:
        result = chain.answer(payload.question)
    except Exception as exc:  # upstream (Mistral) failure
        raise HTTPException(
            status_code=502, detail="The language model backend is unavailable."
        ) from exc
    return AskResponse(**result)


@app.post(
    "/rebuild",
    response_model=RebuildResponse,
    status_code=202,
    summary="Rebuild the index (full refresh)",
)
def rebuild(
    request: Request,
    background: BackgroundTasks,
    _: None = Depends(require_rebuild_token),
) -> RebuildResponse:
    """Trigger a background full refresh: re-collect, re-clean and re-index the events.

    Returns immediately (202). A lock prevents concurrent rebuilds (409 if one is already
    running); when the rebuild finishes, the in-memory retriever is reloaded.
    """
    lock: threading.Lock = request.app.state.rebuild_lock
    chain: RAGChain = request.app.state.rag_chain
    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A rebuild is already in progress.")

    def _run() -> None:
        try:
            full_refresh()
            chain.reload()
        finally:
            lock.release()

    background.add_task(_run)
    return RebuildResponse(status="rebuild scheduled")
