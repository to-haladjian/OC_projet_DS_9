"""Build the committed evaluation corpus from the cleaned dataset (reproducible).

The eval corpus is a small, fixed subset of the events the RAG indexes — committed so the
Ragas evaluation (``evaluate_rag.py``) builds the same tiny FAISS index every run (fast,
deterministic, CI-friendly). With the POC scoped to Hauts-de-Seine (92) the whole cleaned
dataset is one department, so we just take a fixed-seed sample of it.

    poetry run python evaluation/build_corpus.py            # default sample size
    poetry run python evaluation/build_corpus.py --size 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

# Fixed seed so the committed corpus is reproducible (mirrors generate_testset.py).
_SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, default=config.CLEAN_CSV)
    parser.add_argument("--out", type=Path, default=config.EVAL_CORPUS)
    parser.add_argument(
        "--size",
        type=int,
        default=200,
        help="Number of events to sample for the eval corpus (default: 200).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.clean, dtype=str, keep_default_na=False)
    if len(df) > args.size:
        df = df.sample(n=args.size, random_state=_SEED).reset_index(drop=True)
    df = df[config.SCHEMA_COLUMNS]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} events -> {args.out}")


if __name__ == "__main__":
    main()
