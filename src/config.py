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
RAW_CSV = DATA_DIR / "openagenda_92_events.csv"
CLEAN_CSV = DATA_DIR / "events_clean.csv"
FAISS_DIR = PROJECT_ROOT / "faiss_index"

# POC scope: Hauts-de-Seine only (INSEE department code 92). Used to keep only
# Hauts-de-Seine events during cleaning. A frozenset so the ``.isin`` filter and a
# future multi-department scope both stay one-line changes.
TARGET_DEPARTMENT_CODES = frozenset({"92"})

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

# --- RAG chain (two Mistral calls: filter extraction, then generation) ---
# Call 1 (NER): structured-filter extraction — deterministic JSON, so temperature 0.
FILTER_MODEL = "mistral-small-latest"
FILTER_TEMPERATURE = 0.0
# Call 2: answer generation — low temperature for grounded, stable answers.
GENERATION_MODEL = "mistral-small-latest"
GENERATION_TEMPERATURE = 0.1
# Documents retrieved per query (after metadata pre-filtering).
RETRIEVAL_K = 6

# --- Evaluation (Ragas) ---
EVAL_DIR = PROJECT_ROOT / "evaluation"
EVAL_CORPUS = EVAL_DIR / "corpus.csv"  # committed event subset the test set targets
EVAL_TESTSET = EVAL_DIR / "testset.json"  # curated + reviewed-generated Q/R pairs
EVAL_RESULTS_DIR = EVAL_DIR / "results"  # per-run scores (gitignored)
# Committed snapshot of the latest eval run's answers (question/response/reference +
# deterministic scores). Refreshed by scripts/evaluate_rag.py; recomputed by the offline
# CI gate (tests/test_eval_metrics.py) so quality regressions are caught on every PR
# without any Mistral call.
EVAL_ANSWERS_SNAPSHOT = EVAL_DIR / "answers_latest.json"
# Judge model for the Ragas metrics. mistral-small is fast enough to evaluate the whole
# test set within a CI window; throttled concurrency (see scripts/evaluate_rag.py) keeps its
# structured-output calls under the rate limit. Switch to mistral-large-latest for a
# higher-fidelity (but much slower) judge.
EVAL_JUDGE_MODEL = "mistral-small-latest"
# Minimum acceptable mean score per metric (used by scripts/evaluate_rag.py --fail-under / CI).
EVAL_THRESHOLDS = {
    "faithfulness": 0.70,
    "answer_relevancy": 0.70,
    "context_precision": 0.50,
    "context_recall": 0.50,
}

# Deterministic, LLM-free baseline metrics (evaluation/metrics.py). Thresholds are
# deliberately modest: generative answers rephrase the reference, so exact match is rare
# and token-F1 measures lexical overlap, not correctness. The goal is regression
# detection in the offline CI gate (tests/test_eval_metrics.py), not a high bar.
OFFLINE_EVAL_THRESHOLDS = {
    "exact_match": 0.0,
    "token_f1": 0.30,
}


def load_mistral_api_key() -> str:
    """Return the Mistral API key from the environment (loading ``.env`` first)."""
    load_dotenv()
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise SystemExit(
            "MISTRAL_API_KEY is not set (expected in .env or environment)."
        )
    return key
