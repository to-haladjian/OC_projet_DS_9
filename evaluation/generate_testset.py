"""Generate candidate Q/R pairs with Ragas TestsetGenerator (manual, reviewed step).

This produces the *generated* half of the hybrid test set. Run it occasionally, review the
output, and merge the good pairs into ``evaluation/testset.json`` (with
``"source": "generated"``). It is deliberately NOT wired into evaluate_rag.py or CI — the
committed ``testset.json`` is the source of truth.

Building the knowledge graph runs an LLM extractor per source document over the *whole*
corpus (independent of ``--size``), so ``--docs`` caps how many documents are used to keep
cost/time bounded and stay under Mistral's rate limit.

    poetry run python evaluation/generate_testset.py --size 8 --out evaluation/generated_candidates.json
    poetry run python evaluation/generate_testset.py --size 12 --docs 80   # more, over more docs
    poetry run python evaluation/generate_testset.py --docs 0              # use the whole corpus
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Make the project root importable when run as a plain script (so ``evaluation`` and
# ``src`` resolve), then register the ragas/langchain shim BEFORE any ragas import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import evaluation._compat  # noqa: E402,F401  (registers the langchain/ragas shim on import)
from langchain.chat_models import init_chat_model  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.run_config import RunConfig  # noqa: E402
from ragas.testset import TestsetGenerator  # noqa: E402

from src import config  # noqa: E402
from src.indexing.build_index import load_events, make_embeddings  # noqa: E402
from src.indexing.chunking import build_documents  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=config.EVAL_CORPUS)
    parser.add_argument("--size", type=int, default=8, help="Number of pairs to generate.")
    parser.add_argument(
        "--docs",
        type=int,
        default=50,
        help=(
            "Cap the number of source documents fed to the knowledge-graph transforms "
            "(sampled deterministically). The transforms run an LLM call per document, so "
            "this bounds cost/time; 0 = use the whole corpus."
        ),
    )
    parser.add_argument(
        "--out", type=Path, default=config.EVAL_DIR / "generated_candidates.json"
    )
    args = parser.parse_args()

    config.load_mistral_api_key()
    documents = build_documents(load_events(args.corpus))
    if args.docs and len(documents) > args.docs:
        # Deterministic subsample: building the KG runs an LLM extractor per document over
        # the whole corpus regardless of --size, which is slow and trips Mistral's rate
        # limit. A fixed-seed sample keeps it cheap and reproducible.
        documents = random.Random(42).sample(documents, args.docs)
    print(f"Using {len(documents)} documents from {args.corpus}.")

    generator = TestsetGenerator(
        llm=LangchainLLMWrapper(
            init_chat_model(config.EVAL_JUDGE_MODEL, model_provider="mistralai", temperature=0)
        ),
        embedding_model=LangchainEmbeddingsWrapper(make_embeddings()),
    )
    # Throttle concurrency + retry like evaluate_rag.py: the default 16 workers overruns
    # Mistral's rate limit (HTTP 429) during the extractor passes.
    run_config = RunConfig(max_workers=3, timeout=300, max_retries=15, max_wait=90)
    dataset = generator.generate_with_langchain_docs(
        documents, testset_size=args.size, run_config=run_config
    )

    df = dataset.to_pandas()
    candidates = [
        {
            "question": row["user_input"],
            "ground_truth": row.get("reference", ""),
            "category": "generated",
            "source": "generated",
        }
        for _, row in df.iterrows()
    ]
    args.out.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(candidates)} candidate pairs -> {args.out}")
    print("Review them and merge the good ones into evaluation/testset.json.")


if __name__ == "__main__":
    main()
