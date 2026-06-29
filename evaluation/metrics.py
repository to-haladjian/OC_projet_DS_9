"""Deterministic, LLM-free evaluation metrics (exact match + token-level F1).

These complement the Ragas judge metrics with a reference baseline that needs no API
call: given the system answer and the human reference, they score lexical overlap only.
Because they are pure and deterministic, they double as the offline CI gate — the test
suite recomputes them over a committed answers snapshot (see
``tests/test_eval_metrics.py``) on every push/PR, whereas Ragas stays a manual job.

The normalisation is SQuAD-style (lowercase, strip accents/punctuation, collapse
whitespace) so that cosmetic differences ("À 20h00." vs "a 20h00") do not penalise a
correct answer.
"""

from __future__ import annotations

import re
import string
import unicodedata

_PUNCT = str.maketrans({c: " " for c in string.punctuation + "«»–—’“”…"})


def normalize_answer(text: str) -> str:
    """Lowercase, strip accents and punctuation, and collapse whitespace (SQuAD-style)."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold().translate(_PUNCT)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> list[str]:
    return normalize_answer(text).split()


def exact_match(prediction: str, reference: str) -> float:
    """1.0 if the normalized prediction equals the normalized reference, else 0.0."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(reference) else 0.0


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
    """Mean exact_match and token_f1 over ``[{"response", "reference"}, ...]``.

    Returns ``{"exact_match": 0.0, "token_f1": 0.0}`` for an empty list so callers (and
    the CI gate) never divide by zero.
    """
    if not pairs:
        return {"exact_match": 0.0, "token_f1": 0.0}
    n = len(pairs)
    em = sum(exact_match(p["response"], p["reference"]) for p in pairs) / n
    f1 = sum(token_f1(p["response"], p["reference"]) for p in pairs) / n
    return {"exact_match": em, "token_f1": f1}
