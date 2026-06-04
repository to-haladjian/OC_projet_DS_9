"""Build (and load) the FAISS vector store from the cleaned events.

Indexing pipeline: read the cleaned CSV -> render Documents (metadata header + body) ->
chunk -> embed with Mistral (``mistral-embed``) -> index in FAISS -> persist to disk so
the RAG retriever can load it without re-embedding. Embeddings are computed in batches to
stay within Mistral's API rate limits.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings

from src import config
from src.indexing.chunking import build_documents, split_documents


def make_embeddings() -> MistralAIEmbeddings:
    """Instantiate the Mistral embeddings client used for indexing and querying."""
    return MistralAIEmbeddings(
        model=config.EMBED_MODEL,
        api_key=config.load_mistral_api_key(),
    )


def load_events(csv_path: Path) -> pd.DataFrame:
    """Read the cleaned events CSV, keeping location codes as strings (not floats)."""
    return pd.read_csv(
        csv_path, dtype={"postalcode": "string", "department": "string"}
    )


def embed_and_index(
    documents: list[Document],
    embeddings: MistralAIEmbeddings,
    batch_size: int,
) -> FAISS:
    """Embed ``documents`` in batches and index them in a FAISS vector store."""
    total = len(documents)
    if total == 0:
        raise ValueError("No documents to index.")
    print(f"Embedding {total} chunks in batches of {batch_size} ...")

    vector_store = FAISS.from_documents(documents[:batch_size], embeddings)
    print(f"  indexed {min(batch_size, total)}/{total}")

    for start in range(batch_size, total, batch_size):
        batch = documents[start : start + batch_size]
        vector_store.add_documents(batch)
        print(f"  indexed {min(start + batch_size, total)}/{total}")

    return vector_store


def build_index(
    csv_path: Path = config.CLEAN_CSV,
    outdir: Path = config.FAISS_DIR,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    batch_size: int = 64,
    limit: int | None = None,
    no_chunk: bool = False,
) -> FAISS:
    """Run the full indexing pipeline and persist the FAISS store to ``outdir``."""
    print(f"Loading cleaned events from {csv_path} ...")
    df = load_events(csv_path)
    documents = build_documents(df)
    if limit is not None:
        documents = documents[:limit]
    print(f"Built {len(documents)} event documents.")

    if no_chunk:
        chunks = documents
        print("Chunking disabled: indexing each event as a single document.")
    else:
        chunks = split_documents(documents, chunk_size, chunk_overlap)
        print(f"Split into {len(chunks)} chunks.")

    embeddings = make_embeddings()
    vector_store = embed_and_index(chunks, embeddings, batch_size)

    outdir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(outdir))
    print(f"Saved FAISS index -> {outdir}")
    return vector_store


def load_vector_store(
    index_dir: Path = config.FAISS_DIR,
    embeddings: MistralAIEmbeddings | None = None,
) -> FAISS:
    """Load a persisted FAISS store (reused by the RAG chain and the future API)."""
    embeddings = embeddings or make_embeddings()
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        # The FAISS docstore is pickled; safe here because we created the index.
        allow_dangerous_deserialization=True,
    )
