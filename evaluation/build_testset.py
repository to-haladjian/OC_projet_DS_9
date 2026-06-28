"""Build the evaluation test set (reproducible, grounded in the committed corpus).

The test set mixes two sources, mirroring the hybrid approach documented in the README:

* **curated** — a small set of hand-written *topic*, *location_multi* and *edge* cases
  (refusals / out-of-scope) that are awkward to template (see ``SPECIALS``);
* **generated** — *factual_lookup* pairs produced **deterministically** from real events
  of ``evaluation/corpus.csv``. Every question targets one event and its ground-truth
  answer is built from that event's own fields (title / venue / city / daterange /
  conditions), so the reference is grounded by construction — no LLM, no hallucination.

Because the factual pairs are drawn from the same committed corpus the eval index is built
from, every question is answerable, and the whole file is reproducible from a fixed seed.

    poetry run python evaluation/build_testset.py            # default: 100 questions
    poetry run python evaluation/build_testset.py --size 60
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

_SEED = 42

# --- Curated, non-factual cases (kept verbatim; hard to template) -------------------
# Topic / multi-location pairs reference events present in corpus.csv; the edge cases are
# scope-based (refusal / out-of-scope) and hold regardless of the sampled corpus.
SPECIALS: list[dict] = [
    {
        "question": "Quelles visites guidées de médiathèques sont proposées à Clamart ?",
        "ground_truth": "À Clamart, une visite guidée de la Médiathèque La Buanderie est proposée le samedi 20 septembre 2025 à 15h00, sur inscription.",
        "category": "topic_location",
        "source": "curated",
    },
    {
        "question": "Y a-t-il des expositions à Fontenay-aux-Roses ?",
        "ground_truth": "Oui : l'exposition de Pierre Olivier Kohler est présentée à la Médiathèque de Fontenay-aux-Roses, du 3 au 5 octobre 2025.",
        "category": "topic_location",
        "source": "curated",
    },
    {
        "question": "Quels spectacles de cirque ont lieu à Nanterre ?",
        "ground_truth": "À Nanterre, des spectacles de cirque « Les Échappées Cirque » sont proposés dans plusieurs parcs de la ville durant l'été 2025 (entrée gratuite).",
        "category": "topic_location",
        "source": "curated",
    },
    {
        "question": "Quels ateliers de jardinage sont organisés ?",
        "ground_truth": "Un atelier bouturage est organisé au Jardin partagé Hoffmann à Bourg-la-Reine, le dimanche 30 août à 15h00.",
        "category": "topic",
        "source": "curated",
    },
    {
        "question": "Quels événements autour du cinéma sont proposés ?",
        "ground_truth": "Le Ciné d'Issy à Issy-les-Moulineaux propose la projection de « La Vénus électrique », du 17 au 21 juin.",
        "category": "topic",
        "source": "curated",
    },
    {
        "question": "Que peut-on faire à Issy-les-Moulineaux ?",
        "ground_truth": "À Issy-les-Moulineaux, plusieurs événements sont proposés, par exemple la pièce « L'art de perdre » à l'Auditorium Niermayer, la projection « La Vénus électrique » au Ciné d'Issy, et un atelier « Habilitation Gravure laser » au Temps des Cerises.",
        "category": "location_multi",
        "source": "curated",
    },
    {
        "question": "Y a-t-il des compétitions de ski alpin dans les Hauts-de-Seine ?",
        "ground_truth": "Non, je n'ai pas trouvé d'événement correspondant à des compétitions de ski alpin dans les données.",
        "category": "edge_unknown",
        "source": "curated",
    },
    {
        "question": "Quels événements sont prévus à Marseille ?",
        "ground_truth": "Aucun événement à Marseille n'est disponible : les données ne couvrent que le département des Hauts-de-Seine (92).",
        "category": "edge_unknown",
        "source": "curated",
    },
    {
        "question": "Où puis-je assister à un Grand Prix de Formule 1 dans les Hauts-de-Seine ?",
        "ground_truth": "Aucun Grand Prix de Formule 1 n'est proposé : il n'y a pas d'événement de ce type dans les Hauts-de-Seine dans les données.",
        "category": "edge_outofscope",
        "source": "curated",
    },
]


def _is_usable(row: pd.Series) -> bool:
    """Keep events with the clean, complete fields a templated question needs."""
    title = row["title"].strip()
    return bool(
        title
        and row["venue"].strip()
        and row["city"].strip()
        and row["daterange"].strip()
        and len(title) < 70
        and not title.upper().startswith("ANNUL")
        and title.lower() != row["city"].strip().lower()
    )


def _factual_pair(row: pd.Series, variant: int) -> dict:
    """Build one grounded factual question/answer from an event, varying the phrasing."""
    title, venue, city = row["title"].strip(), row["venue"].strip(), row["city"].strip()
    when, conditions = row["daterange"].strip(), row["conditions"].strip()
    is_free = "gratuit" in conditions.lower()

    if variant == 0:
        q = f"Où et quand a lieu « {title} » ?"
        a = f"« {title} » a lieu à {venue} à {city}, {when}."
    elif variant == 1:
        q = f"Dans quelle ville se déroule « {title} » ?"
        a = f"« {title} » se déroule à {city} (au lieu : {venue})."
    elif variant == 2:
        q = f"Quand a lieu « {title} » ?"
        a = f"« {title} » a lieu : {when} (à {venue}, {city})."
    elif variant == 3 and is_free:
        q = f"L'entrée à « {title} » est-elle gratuite ?"
        a = f"Oui, l'entrée à « {title} » ({venue}, {city}) est gratuite."
    else:  # variant 3 without a free hint, or any fallback
        q = f"Dans quel lieu se tient « {title} » à {city} ?"
        a = f"« {title} » se tient à {venue}, à {city}, {when}."

    return {"question": q, "ground_truth": a, "category": "factual_lookup", "source": "generated"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=config.EVAL_CORPUS)
    parser.add_argument("--out", type=Path, default=config.EVAL_TESTSET)
    parser.add_argument(
        "--size", type=int, default=100, help="Total number of Q/R pairs (default: 100)."
    )
    args = parser.parse_args()

    df = pd.read_csv(args.corpus, dtype=str, keep_default_na=False)
    usable = df[df.apply(_is_usable, axis=1)].drop_duplicates(subset="title")

    n_factual = max(args.size - len(SPECIALS), 0)
    if len(usable) < n_factual:
        raise SystemExit(
            f"Only {len(usable)} usable events for {n_factual} factual questions; "
            "lower --size or grow the corpus (evaluation/build_corpus.py)."
        )

    rows = usable.to_dict("records")
    random.Random(_SEED).shuffle(rows)
    factual = [_factual_pair(pd.Series(r), i % 4) for i, r in enumerate(rows[:n_factual])]

    testset = SPECIALS + factual
    args.out.write_text(
        json.dumps(testset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {len(testset)} pairs -> {args.out} "
        f"({len(SPECIALS)} curated + {len(factual)} generated factual)"
    )


if __name__ == "__main__":
    main()
