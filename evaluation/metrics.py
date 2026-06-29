"""Reference-based evaluation metrics that do not need an LLM judge.

These complement the Ragas judge metrics with a cheaper, reproducible baseline:

* ``token_f1`` — lexical overlap between the answer and the reference. Pure text, so it
  is **recomputed offline** by the test suite over the committed answers snapshot (see
  ``tests/test_eval_metrics.py``) on every push / PR — the per-PR quality gate.
* ``cosine_similarity`` — semantic closeness of two embedding vectors. The text→vector
  step needs the embedding API, so ``answer_similarity`` is computed **when the eval
  runs** (``scripts/evaluate_rag.py``) and stored in the snapshot; the offline gate reads
  that recorded value. Embeddings are deterministic, so the score is reproducible.

The token normalisation is SQuAD-style (lowercase, strip accents/punctuation, collapse
whitespace) so cosmetic differences ("À 20h00." vs "a 20h00") do not penalise an answer.
"""

from __future__ import annotations

import math
import re
import string
import unicodedata
from typing import Sequence

_PUNCT = str.maketrans({c: " " for c in string.punctuation + "«»–—’“”…"})


def normalize_answer(text: str) -> str:
    """Lowercase, strip accents and punctuation, and collapse whitespace (SQuAD-style)."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold().translate(_PUNCT)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> list[str]:
    return normalize_answer(text).split()


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors (0.0 if either is empty or zero-norm).

    Pure vector math (no API): the embeddings are produced upstream by the model and
    passed in here. Range is [-1, 1]; for sentence embeddings of related French text it
    sits well above 0.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def token_f1(prediction: str, reference: str) -> float:
    """Token-overlap F1 between prediction and reference (0.0-1.0).

    Multiplicity-aware (like the SQuAD F1): a token counts as a match as many times as it
    appears in both. Returns 1.0 when both sides are empty and 0.0 when exactly one is.
    """
    pred, ref = _tokens(prediction), _tokens(reference)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0

    common = 0
    ref_counts: dict[str, int] = {}
    for tok in ref:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1
    for tok in pred:
        if ref_counts.get(tok, 0) > 0:
            common += 1
            ref_counts[tok] -= 1

    if common == 0:
        return 0.0
    precision = common / len(pred)
    recall = common / len(ref)
    return 2 * precision * recall / (precision + recall)


def score_pairs(pairs: list[dict]) -> dict[str, float]:
    """Mean deterministic scores over ``[{"response", "reference", ...}, ...]``.

    ``token_f1`` is **recomputed** from the text. ``answer_similarity`` is **read** from a
    pre-stored ``"answer_similarity"`` field when present (it cannot be recomputed without
    the embedding API) and omitted otherwise. Returns ``{"token_f1": 0.0}`` for an empty
    list so callers (and the CI gate) never divide by zero.
    """
    if not pairs:
        return {"token_f1": 0.0}
    n = len(pairs)
    means = {"token_f1": sum(token_f1(p["response"], p["reference"]) for p in pairs) / n}
    sims = [p["answer_similarity"] for p in pairs if "answer_similarity" in p]
    if sims:
        means["answer_similarity"] = sum(sims) / len(sims)
    return means
