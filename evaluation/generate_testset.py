"""Generate candidate Q/R pairs with Ragas TestsetGenerator (manual, reviewed step).

This produces the *generated* half of the hybrid test set. Run it occasionally, review the
output, and merge the good pairs into ``evaluation/testset.json`` (with
``"source": "generated"``). It is deliberately NOT wired into evaluate_rag.py or CI — the
committed ``testset.json`` is the source of truth.

    poetry run python evaluation/generate_testset.py --size 8 --out evaluation/generated_candidates.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import evaluation._compat  # noqa: F401  (registers the langchain/ragas shim on import)
from langchain.chat_models import init_chat_model
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator

from src import config
from src.indexing.build_index import load_events, make_embeddings
from src.indexing.chunking import build_documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=config.EVAL_CORPUS)
    parser.add_argument("--size", type=int, default=8, help="Number of pairs to generate.")
    parser.add_argument(
        "--out", type=Path, default=config.EVAL_DIR / "generated_candidates.json"
    )
    args = parser.parse_args()

    config.load_mistral_api_key()
    documents = build_documents(load_events(args.corpus))
    print(f"Loaded {len(documents)} documents from {args.corpus}.")

    generator = TestsetGenerator(
        llm=LangchainLLMWrapper(
            init_chat_model(config.EVAL_JUDGE_MODEL, model_provider="mistralai", temperature=0)
        ),
        embedding_model=LangchainEmbeddingsWrapper(make_embeddings()),
    )
    dataset = generator.generate_with_langchain_docs(documents, testset_size=args.size)

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
