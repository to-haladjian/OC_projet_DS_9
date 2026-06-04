"""Build a FAISS vector store from the collected OpenAgenda Île-de-France events.

This follows the LangChain RAG indexing pattern
(https://docs.langchain.com/oss/python/langchain/rag): turn the source data into
``Document`` objects, split them into chunks, embed the chunks and index them in a
vector store. Here the source is the cleaned CSV produced by ``clean_events.py``, the
embeddings come from Mistral (``mistral-embed``), and the store is FAISS, persisted
to disk so the RAG retriever can load it without re-embedding.

Embeddings are computed in batches to stay within Mistral's API rate limits.

Usage:
    poetry run python scripts/build_vector_store.py
    poetry run python scripts/build_vector_store.py --csv data/openagenda_idf_events.csv \\
        --outdir faiss_index --batch-size 64
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make ``src`` importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config

PROJECT_ROOT = config.PROJECT_ROOT

# Column holding the rich, free-text description we embed for semantic search,
# and the metadata columns carried alongside each vector (for citations/filtering).
CONTENT_FIELD = config.CONTENT_FIELD
METADATA_FIELDS = config.METADATA_FIELDS


def load_documents(csv_path: Path) -> list[Document]:
    """Read the cleaned events CSV and turn each event into a LangChain ``Document``.

    Only events with a non-empty description are kept (that is the text we embed).
    The title is prepended to the body so it is part of what gets indexed.
    """
    # Keep the location codes as strings (CSV would otherwise re-infer them as floats),
    # so they stay clean in the FAISS metadata.
    df = pd.read_csv(csv_path, dtype={"postalcode": "string", "department": "string"})
    df = df[df[CONTENT_FIELD].notna()]

    documents: list[Document] = []
    for row in df.to_dict(orient="records"):
        title = row.get("title")
        body = str(row[CONTENT_FIELD])
        page_content = f"{title}\n\n{body}" if pd.notna(title) else body

        metadata = {
            field: row[field]
            for field in METADATA_FIELDS
            if field in row and pd.notna(row[field])
        }
        documents.append(Document(page_content=page_content, metadata=metadata))

    return documents


def split_documents(
    documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Split documents into overlapping chunks for finer-grained retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)


def build_vector_store(
    documents: list[Document],
    embeddings: MistralAIEmbeddings,
    batch_size: int,
) -> FAISS:
    """Embed ``documents`` in batches and index them in a FAISS vector store."""
    total = len(documents)
    print(f"Embedding {total} chunks in batches of {batch_size} ...")

    first = documents[:batch_size]
    vector_store = FAISS.from_documents(first, embeddings)
    print(f"  indexed {min(batch_size, total)}/{total}")

    for start in range(batch_size, total, batch_size):
        batch = documents[start : start + batch_size]
        vector_store.add_documents(batch)
        print(f"  indexed {min(start + batch_size, total)}/{total}")

    return vector_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=config.CLEAN_CSV,
        help="Path to the cleaned events CSV (default: <project>/data/events_clean.csv).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=config.FAISS_DIR,
        help="Directory to save the FAISS index into (default: <project>/faiss_index).",
    )
    parser.add_argument(
        "--no-chunk",
        action="store_true",
        help="Index each event as a single document instead of splitting it into "
        "chunks (descriptions are short, so this is often fine).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Max characters per chunk (default: 1000). Ignored with --no-chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Character overlap between chunks (default: 200). Ignored with --no-chunk.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Chunks per embedding API call (default: 64).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only index the first N events (handy for a quick test run).",
    )
    args = parser.parse_args()

    load_dotenv()
    if "MISTRAL_API_KEY" not in os.environ:
        raise SystemExit("MISTRAL_API_KEY is not set (expected in .env or environment).")

    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.environ["MISTRAL_API_KEY"],
    )

    print(f"Loading events from {args.csv} ...")
    documents = load_documents(args.csv)
    if args.limit is not None:
        documents = documents[: args.limit]
    print(f"Loaded {len(documents)} events with a long description.")

    if args.no_chunk:
        chunks = documents
        print("Chunking disabled: indexing each event as a single document.")
    else:
        chunks = split_documents(documents, args.chunk_size, args.chunk_overlap)
        print(f"Split into {len(chunks)} chunks.")

    vector_store = build_vector_store(chunks, embeddings, args.batch_size)

    args.outdir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(args.outdir))
    print(f"Saved FAISS index -> {args.outdir}")


if __name__ == "__main__":
    main()
