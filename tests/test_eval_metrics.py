"""Tests for the deterministic eval metrics + the offline CI quality gate (no API).

The unit tests pin the metric behaviour; the regression test recomputes the metrics over
the committed answers snapshot (``evaluation/answers_latest.json``) and asserts they meet
the configured thresholds. It skips when the snapshot is absent, so the suite never breaks
before the first ``scripts/evaluate_rag.py`` run produces one.
"""

from __future__ import annotations

import json

import pytest

from evaluation.metrics import exact_match, normalize_answer, score_pairs, token_f1
from src import config


# --- normalize_answer ---

def test_normalize_strips_accents_case_and_punctuation():
    assert normalize_answer("À 20H00, au Théâtre !") == "a 20h00 au theatre"


def test_normalize_collapses_whitespace():
    assert normalize_answer("  deux\tmots  ") == "deux mots"


# --- exact_match ---

def test_exact_match_ignores_cosmetic_differences():
    assert exact_match("La Vénus électrique.", "la venus electrique") == 1.0


def test_exact_match_rejects_different_answers():
    assert exact_match("Nanterre", "Antony") == 0.0


# --- token_f1 ---

def test_token_f1_is_one_for_equivalent_answers():
    assert token_f1("Concert à Nanterre", "concert a nanterre") == 1.0


def test_token_f1_is_zero_without_overlap():
    assert token_f1("Concert jazz Nanterre", "Exposition peinture Meudon") == 0.0


def test_token_f1_is_partial_for_partial_overlap():
    score = token_f1("concert jazz nanterre", "concert nanterre")
    assert 0.0 < score < 1.0


def test_token_f1_handles_empty_inputs():
    assert token_f1("", "") == 1.0
    assert token_f1("quelque chose", "") == 0.0


# --- score_pairs ---

def test_score_pairs_averages_over_pairs():
    pairs = [
        {"response": "concert a nanterre", "reference": "Concert à Nanterre"},  # EM 1
        {"response": "exposition meudon", "reference": "Concert à Nanterre"},  # EM 0
    ]
    means = score_pairs(pairs)
    assert means["exact_match"] == 0.5
    assert 0.0 <= means["token_f1"] <= 1.0


def test_score_pairs_empty_is_zero():
    assert score_pairs([]) == {"exact_match": 0.0, "token_f1": 0.0}


# --- Offline regression gate over the committed snapshot ---

def test_committed_snapshot_meets_offline_thresholds():
    snapshot_path = config.EVAL_ANSWERS_SNAPSHOT
    if not snapshot_path.exists():
        pytest.skip(
            "No answers snapshot yet; run scripts/evaluate_rag.py to generate "
            f"{snapshot_path.name} and activate the offline gate."
        )
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot, "answers snapshot is empty"

    means = score_pairs(snapshot)
    for name, threshold in config.OFFLINE_EVAL_THRESHOLDS.items():
        assert means[name] >= threshold, (
            f"{name}={means[name]:.3f} below threshold {threshold}"
        )
