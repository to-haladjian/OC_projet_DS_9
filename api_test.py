"""Manual functional smoke test for the running API (real Mistral calls).

Start the server first, then run this script:

    poetry run uvicorn src.api.main:app
    poetry run python api_test.py
    poetry run python api_test.py --url http://localhost:8000 "Des expositions à Paris ?"
"""

from __future__ import annotations

import argparse

import httpx

DEFAULT_QUESTION = "Quels concerts de jazz à Paris ce week-end ?"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    with httpx.Client(base_url=args.url, timeout=120) as client:
        health = client.get("/health")
        print(f"GET /health -> {health.status_code} {health.json()}")

        resp = client.post("/ask", json={"question": args.question})
        print(f"\nPOST /ask -> {resp.status_code}")
        resp.raise_for_status()
        body = resp.json()
        print(f"\nFiltres : {body['filters']}")
        print(f"\nRéponse :\n{body['answer']}")
        print("\nSources :")
        for meta in body["sources"]:
            print(
                f"  - {meta.get('title', '?')} | {meta.get('city', '')} "
                f"| {meta.get('daterange', '')} | {meta.get('url', '')}"
            )


if __name__ == "__main__":
    main()
