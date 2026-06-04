"""Turn cleaned event rows into LangChain ``Document`` objects, then chunk them.

Each event is rendered with a short metadata header (title / date / venue) prepended to
its description before embedding. Anchoring the embedding on those fields makes semantic
search match on *what / when / where* even when the free-text description is terse. The
same fields are also stored as ``Document.metadata`` so they can be cited back to the
user and used for pre-filtering in the RAG chain.
"""

from __future__ import annotations

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config


def _format_header(row: dict) -> str:
    """Build the 'Titre / Date / Lieu' header prepended to each event's description."""
    title = row.get("title")
    daterange = row.get("daterange")
    venue = row.get("venue")
    city = row.get("city")
    postalcode = row.get("postalcode")

    lines: list[str] = []
    if pd.notna(title):
        lines.append(f"Titre: {title}")
    if pd.notna(daterange):
        lines.append(f"Date: {daterange}")

    location = ", ".join(str(v) for v in (venue, city) if pd.notna(v))
    if pd.notna(postalcode):
        location = f"{location} ({postalcode})" if location else str(postalcode)
    if location:
        lines.append(f"Lieu: {location}")

    return "\n".join(lines)


def build_event_document(row: dict) -> Document:
    """Render one cleaned event row into a ``Document`` (header + description + metadata)."""
    header = _format_header(row)
    body = str(row[config.CONTENT_FIELD])
    page_content = f"{header}\n\n{body}" if header else body

    metadata = {
        field: row[field]
        for field in config.METADATA_FIELDS
        if field in row and pd.notna(row[field])
    }
    return Document(page_content=page_content, metadata=metadata)


def build_documents(df: pd.DataFrame) -> list[Document]:
    """Turn every event with a non-empty description into a ``Document``."""
    df = df[df[config.CONTENT_FIELD].notna()]
    return [build_event_document(row) for row in df.to_dict(orient="records")]


def split_documents(
    documents: list[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into overlapping chunks for finer-grained retrieval.

    Metadata is copied onto every chunk by the splitter, so pre-filtering and citations
    keep working at the chunk level.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(documents)
