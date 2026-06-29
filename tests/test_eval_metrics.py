"""Tests for the deterministic eval metrics + the offline CI quality gate (no API).

The unit tests pin the metric behaviour; the regression test recomputes the metrics over
the committed answers snapshot (``evaluation/answers_latest.json``) and asserts they meet
the configured thresholds. It skips when the snapshot is absent, so the suite never breaks
before the first ``scripts/evaluate_rag.py`` run produces one.
"""

from __future__ import annotations

import json

import pytest

from evaluation.metrics import (
    cosine_similarity,
    normalize_answer,
    score_pairs,
    token_f1,
)
from src import config


# --- normalize_answer ---

def test_normalize_strips_accents_case_and_punctuation():
    assert normalize_answer("À 20H00, au Théâtre !") == "a 20h00 au theatre"


def test_normalize_collapses_whitespace():
    assert normalize_answer("  deux\tmots  ") == "deux mots"


# --- cosine_similarity ---

def test_cosine_similarity_identical_vectors_is_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_opposite_is_minus_one():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_is_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


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

def test_score_pairs_recomputes_token_f1_and_reads_similarity():
    pairs = [
        {"response": "concert a nanterre", "reference": "Concert à Nanterre",
         "answer_similarity": 0.9},
        {"response": "exposition meudon", "reference": "Concert à Nanterre",
         "answer_similarity": 0.5},
    ]
    means = score_pairs(pairs)
    assert 0.0 <= means["token_f1"] <= 1.0
    assert means["answer_similarity"] == pytest.approx(0.7)  # read, not recomputed


def test_score_pairs_omits_similarity_when_absent():
    means = score_pairs([{"response": "a", "reference": "a"}])
    assert "answer_similarity" not in means


def test_score_pairs_empty_is_zero():
    assert score_pairs([]) == {"token_f1": 0.0}


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
