"""Central configuration: paths, schema constants and secrets loading.

Keeping these in one place avoids the duplication that crept into the standalone
``scripts/`` and gives every module (collection, cleaning, indexing, RAG) a single
source of truth for where data lives and which columns matter.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = the directory above ``src/`` so paths resolve regardless of the
# current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_JSON = DATA_DIR / "openagenda_idf_events.json"
RAW_CSV = DATA_DIR / "openagenda_idf_events.csv"
CLEAN_CSV = DATA_DIR / "events_clean.csv"
FAISS_DIR = PROJECT_ROOT / "faiss_index"

# Île-de-France department codes: Paris (75) + petite couronne (92/93/94) +
# grande couronne (77/78/91/95). Used to keep only IDF events during cleaning.
IDF_DEPARTMENT_CODES = frozenset({"75", "77", "78", "91", "92", "93", "94", "95"})

# Columns of the cleaned dataset (output of ``src.data.clean.clean``).
SCHEMA_COLUMNS = [
    "id",
    "title",
    "description",
    "summary",
    "date_start",
    "date_end",
    "daterange",
    "venue",
    "address",
    "city",
    "postalcode",
    "department",
    "coordinates",
    "keywords",
    "conditions",
    "url",
    "updatedat",
]

# Field embedded for semantic search, and columns carried as FAISS metadata
# (used for citations back to the user and for metadata pre-filtering later).
CONTENT_FIELD = "description"
METADATA_FIELDS = [
    "id",
    "title",
    "daterange",
    "date_start",
    "date_end",
    "venue",
    "city",
    "postalcode",
    "department",
    "keywords",
    "url",
]

# --- Indexing ---
# Chunking: large window deliberately (event descriptions are short, often a single
# chunk) — preserves event coherence while still satisfying the chunking requirement.
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 100
EMBED_MODEL = "mistral-embed"


def load_mistral_api_key() -> str:
    """Return the Mistral API key from the environment (loading ``.env`` first)."""
    load_dotenv()
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise SystemExit(
            "MISTRAL_API_KEY is not set (expected in .env or environment)."
        )
    return key
