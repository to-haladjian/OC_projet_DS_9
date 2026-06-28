"""Evaluate the RAG system with Ragas (faithfulness, answer relevancy, context P/R).

For each question in ``evaluation/testset.json`` the RAG chain is run to collect the answer
and the retrieved contexts; Ragas then scores them against the reference answer using a
Mistral judge LLM + ``mistral-embed``. By default the chain runs over a small index built
from the committed ``evaluation/corpus.csv`` (fast, deterministic, CI-friendly); pass
``--index faiss_index`` to evaluate against the full local store instead.

    poetry run python scripts/evaluate_rag.py                 # full test set, eval corpus
    poetry run python scripts/evaluate_rag.py --sample 3      # quick/cheap subset
    poetry run python scripts/evaluate_rag.py --index faiss_index   # full corpus
    poetry run python scripts/evaluate_rag.py --fail-under    # CI gate (exit 1 if below thresholds)
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain.chat_models import init_chat_model

# Make ``src`` importable and register the ragas/langchain compatibility shim BEFORE any
# ragas import happens.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import evaluation._compat  # noqa: E402,F401

from ragas import EvaluationDataset, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.run_config import RunConfig  # noqa: E402
from ragas.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src import config  # noqa: E402
from src.indexing.build_index import build_index  # noqa: E402
from src.rag.chain import RAGChain  # noqa: E402

# answer_relevancy asks the judge for ``strictness`` generations in a single call; with
# ChatMistralAI that response aggregation raises ``TypeError: dict += dict``. One generation
# sidesteps the bug (scores are effectively unchanged).
answer_relevancy.strictness = 1

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


def load_testset(path: Path, sample: int | None) -> list[dict]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    return entries[:sample] if sample else entries


def _answer_with_retry(chain: RAGChain, question: str, max_retries: int = 6) -> dict:
    """Call the chain, backing off on Mistral 429s.

    Unlike the Ragas scoring phase (throttled via RunConfig), this loop fires 2 LLM calls
    per question; over a large test set that bursts past Mistral's rate limit. Exponential
    backoff keeps the whole test set runnable without dropping questions.
    """
    delay = 5.0
    for attempt in range(max_retries + 1):
        try:
            return chain.answer(question, return_contexts=True)
        except Exception as exc:  # noqa: BLE001 - retry only on rate limits
            if "429" not in str(exc) or attempt == max_retries:
                raise
            print(f"      rate-limited, retrying in {delay:.0f}s ...")
            time.sleep(delay)
            delay = min(delay * 2, 90)
    raise RuntimeError("unreachable")


def run_chain(chain: RAGChain, entries: list[dict]) -> EvaluationDataset:
    """Run the RAG chain on each question, collecting Ragas-shaped samples."""
    samples = []
    for i, entry in enumerate(entries, 1):
        question = entry["question"]
        print(f"  [{i}/{len(entries)}] {question}")
        result = _answer_with_retry(chain, question)
        samples.append(
            {
                "user_input": question,
                "retrieved_contexts": result["contexts"],
                "response": result["answer"],
                "reference": entry["ground_truth"],
            }
        )
    return EvaluationDataset.from_list(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help="FAISS index to evaluate against. Default: build a temp index from the eval corpus.",
    )
    parser.add_argument("--testset", type=Path, default=config.EVAL_TESTSET)
    parser.add_argument("--corpus", type=Path, default=config.EVAL_CORPUS)
    parser.add_argument("--sample", type=int, default=None, help="Only evaluate the first N questions.")
    parser.add_argument(
        "--fail-under",
        action="store_true",
        help="Exit non-zero if any metric's mean is below its configured threshold.",
    )
    args = parser.parse_args()

    config.load_mistral_api_key()
    entries = load_testset(args.testset, args.sample)
    print(f"Evaluating {len(entries)} questions from {args.testset.name}.")

    tmpdir: tempfile.TemporaryDirectory | None = None
    if args.index is not None:
        index_dir = args.index
    else:
        tmpdir = tempfile.TemporaryDirectory(prefix="faiss_eval_")
        index_dir = Path(tmpdir.name) / "index"
        print(f"Building eval index from {args.corpus} ...")
        build_index(csv_path=args.corpus, outdir=index_dir)

    chain = RAGChain(index_dir=index_dir)

    print("Running the RAG chain on the test set ...")
    dataset = run_chain(chain, entries)

    judge = LangchainLLMWrapper(
        init_chat_model(config.EVAL_JUDGE_MODEL, model_provider="mistralai", temperature=0)
    )
    embeddings = LangchainEmbeddingsWrapper(chain.embeddings)

    print(f"Scoring with Ragas (judge: {config.EVAL_JUDGE_MODEL}) ...")
    # Throttle concurrency + retry: Ragas fires one job per (sample × metric); at the default
    # 16 workers Mistral returns HTTP 429 and metrics come back NaN. Few workers with retries
    # keep the judge calls under the rate limit.
    run_config = RunConfig(max_workers=3, timeout=300, max_retries=15, max_wait=90)
    result = evaluate(
        dataset, metrics=METRICS, llm=judge, embeddings=embeddings, run_config=run_config
    )
    scores_df = result.to_pandas()

    if tmpdir is not None:
        tmpdir.cleanup()

    # --- Report ---
    metric_cols = [c for c in scores_df.columns if c in {m.name for m in METRICS}]
    means = {c: float(scores_df[c].mean(skipna=True)) for c in metric_cols}

    print("\n=== Ragas scores (mean over the test set) ===")
    for name, value in means.items():
        threshold = config.EVAL_THRESHOLDS.get(name)
        flag = "" if threshold is None else ("  OK" if value >= threshold else "  BELOW")
        print(f"  {name:<20} {value:.3f}{flag}")

    # --- Persist results ---
    config.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    scores_df.to_csv(config.EVAL_RESULTS_DIR / f"results_{stamp}.csv", index=False)
    (config.EVAL_RESULTS_DIR / f"summary_{stamp}.json").write_text(
        json.dumps({"means": means, "n": len(entries)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved detailed results -> {config.EVAL_RESULTS_DIR}/results_{stamp}.csv")

    # --- CI gate ---
    if args.fail_under:
        below = {
            n: v
            for n, v in means.items()
            if config.EVAL_THRESHOLDS.get(n) is not None and v < config.EVAL_THRESHOLDS[n]
        }
        if below:
            print(f"\nFAIL: metrics below threshold: {below}")
            sys.exit(1)
        print("\nPASS: all metrics meet their thresholds.")


if __name__ == "__main__":
    main()
