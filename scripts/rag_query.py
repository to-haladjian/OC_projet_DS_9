"""CLI: answer a question about Île-de-France events using the RAG chain.

Thin wrapper around :class:`src.rag.chain.RAGChain` — the two-call pipeline (filter
extraction -> metadata-filtered FAISS retrieval -> grounded generation). Requires the
FAISS index built by ``scripts/build_vector_store.py``.

Usage:
    python scripts/rag_query.py "Quels concerts de jazz à Paris ce week-end ?"
    python scripts/rag_query.py --k 8 "Des expositions en Seine-Saint-Denis ?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``src`` importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.rag.chain import RAGChain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="The question to ask about the events.")
    parser.add_argument(
        "--index",
        type=Path,
        default=config.FAISS_DIR,
        help=f"Directory of the saved FAISS index (default: {config.FAISS_DIR}).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=config.RETRIEVAL_K,
        help=f"Number of events to retrieve per search (default: {config.RETRIEVAL_K}).",
    )
    args = parser.parse_args()

    chain = RAGChain(index_dir=args.index, k=args.k)
    result = chain.answer(args.query)

    print(f"Filtres extraits : {result['filters']}")
    print(f"\nRéponse :\n{result['answer']}")
    print("\nSources :")
    for meta in result["sources"]:
        title = meta.get("title", "?")
        city = meta.get("city", "")
        daterange = meta.get("daterange", "")
        url = meta.get("url", "")
        print(f"  - {title} | {city} | {daterange} | {url}")


if __name__ == "__main__":
    main()
