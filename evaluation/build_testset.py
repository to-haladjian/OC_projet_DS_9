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

    poetry run python evaluation/build_testset.py            # ~40 curated + 60 generated
    poetry run python evaluation/build_testset.py --factual 30
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
# These are the hard cases the generated factual pairs cannot cover: broad *topic* and
# *topic_location* questions (where retrieval must rank, not just look up), *location_multi*
# questions (several events in one city), and *edge* cases the system must refuse (subject
# absent from the corpus, or out of the 92 scope). Topic / multi-location ground truths
# reference real events of evaluation/corpus.csv; edge cases are scope-based and hold
# regardless of the sampled corpus. Refusals share a common shape so a correct system
# answer is recognisable.
SPECIALS: list[dict] = [
    # --- topic_location: a subject in a given city -------------------------------------
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
        "question": "Quelles pièces de théâtre sont jouées à Châtenay-Malabry ?",
        "ground_truth": "Au Théâtre Firmin Gémier / La Piscine à Châtenay-Malabry, on peut voir « Les Soeurs Hilton » les 7 et 8 octobre 2025 et « JE ME SOUVIENS DE LA TERRE » les 10 et 11 mars.",
        "category": "topic_location",
        "source": "curated",
    },
    {
        "question": "Y a-t-il des expositions à voir à Meudon ?",
        "ground_truth": "Oui : le Musée d'art et d'histoire de Meudon propose une « Visite Flash » de son exposition le dimanche 7 juin (à 15h00 et 16h00, gratuit).",
        "category": "topic_location",
        "source": "curated",
    },
    {
        "question": "Quels concerts sont prévus à Courbevoie ?",
        "ground_truth": "À Courbevoie, le concert pop de Pier di Gia est proposé chez Rémi (Apparemment C) le samedi 20 juin à 19h00.",
        "category": "topic_location",
        "source": "curated",
    },
    {
        "question": "Quels spectacles de théâtre a-t-on à Clichy ?",
        "ground_truth": "À Clichy, « Entrée des artistes » est jouée au Conservatoire Léo Delibes le vendredi 27 mars à 20h30, et le « Contours Festival » se tient au Pavillon Vendôme les 26 et 27 septembre 2025.",
        "category": "topic_location",
        "source": "curated",
    },
    {
        "question": "Quelles expositions découvrir à Issy-les-Moulineaux ?",
        "ground_truth": "À Issy-les-Moulineaux, l'exposition « Shibori, l'indigo révélé » est présentée au Temps des Cerises du 5 décembre 2025 au 25 janvier 2026.",
        "category": "topic_location",
        "source": "curated",
    },
    {
        "question": "Que peut-on visiter dans les musées de Saint-Cloud ?",
        "ground_truth": "À Saint-Cloud, le Musée des Avelines propose « Broderie royale » le samedi 20 septembre 2025 à 15h00, et la scène ouverte « Open Zik » a lieu au Carré (gratuit).",
        "category": "topic_location",
        "source": "curated",
    },
    {
        "question": "Y a-t-il des événements dans les musées de Boulogne-Billancourt ?",
        "ground_truth": "Oui : à Boulogne-Billancourt, le Musée départemental Albert-Kahn propose le festival photographique « Mondes en commun » le samedi 23 mai à 15h00, et le Musée Paul Belmondo accueille « Lire les yeux fermés » le dimanche 7 juin à 16h00.",
        "category": "topic_location",
        "source": "curated",
    },
    {
        "question": "Quelles expositions sont proposées à Sceaux ?",
        "ground_truth": "À Sceaux, le Château de Sceaux (musée départemental) propose la visite libre et gratuite de l'exposition « Trésors et coulisses » les 20 et 21 septembre 2025, et la Bibliothèque municipale présente « Si les Blagis m'étaient contés ».",
        "category": "topic_location",
        "source": "curated",
    },
    {
        "question": "Quels spectacles de danse sont proposés à Gennevilliers ?",
        "ground_truth": "À Gennevilliers, un « Parcours improvisation chorégraphique » est organisé à l'École Paul Langevin, du 30 mars au 3 avril.",
        "category": "topic_location",
        "source": "curated",
    },
    # --- topic: a cross-cutting subject (no city) --------------------------------------
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
        "question": "Quelles pièces de théâtre peut-on voir dans les Hauts-de-Seine ?",
        "ground_truth": "Plusieurs pièces sont programmées, par exemple « Les Soeurs Hilton » au Théâtre Firmin Gémier / La Piscine à Châtenay-Malabry (7 et 8 octobre 2025) et « Entrée des artistes » au Conservatoire Léo Delibes à Clichy (27 mars).",
        "category": "topic",
        "source": "curated",
    },
    {
        "question": "Y a-t-il des spectacles de danse ?",
        "ground_truth": "Oui : le spectacle « ASSEMBLÉE » est donné Place de la cathédrale à Nanterre (7 juin - 6 juillet 2025), et un parcours d'improvisation chorégraphique a lieu à l'École Paul Langevin à Gennevilliers.",
        "category": "topic",
        "source": "curated",
    },
    {
        "question": "Quelles expositions sont à voir en ce moment ?",
        "ground_truth": "Parmi les expositions : « Shibori, l'indigo révélé » au Temps des Cerises à Issy-les-Moulineaux (5 décembre 2025 - 25 janvier 2026) et la « Visite Flash » du Musée d'art et d'histoire de Meudon (dimanche 7 juin, gratuit).",
        "category": "topic",
        "source": "curated",
    },
    {
        "question": "Quels concerts de musique sont programmés ?",
        "ground_truth": "Côté musique, la scène ouverte « Open Zik » de l'ECLA se tient au Carré à Saint-Cloud (gratuit) et un concert pop de Pier di Gia est proposé chez Rémi à Courbevoie le samedi 20 juin à 19h00.",
        "category": "topic",
        "source": "curated",
    },
    {
        "question": "Quels ateliers autour de l'emploi et du recrutement sont proposés ?",
        "ground_truth": "Plusieurs ateliers emploi sont proposés par les agences France Travail, par exemple « Découvrez le secteur aéroportuaire » à Boulogne-Billancourt (jeudi 11 juin) et un atelier « Découverte des métiers de l'industrie » à l'agence de Colombes (lundi 22 juin).",
        "category": "topic",
        "source": "curated",
    },
    {
        "question": "Quelles activités autour de la nature sont organisées ?",
        "ground_truth": "Côté nature, « Regards sur les arbres du parc du Vieux cimetière » a lieu à Courbevoie (samedi 6 juin, 10h00) et « La forêt s'éveille sur les coteaux » est proposé Place Henri Brousse à Meudon.",
        "category": "topic",
        "source": "curated",
    },
    # --- location_multi: several events in one city ------------------------------------
    {
        "question": "Que peut-on faire à Issy-les-Moulineaux ?",
        "ground_truth": "À Issy-les-Moulineaux, plusieurs événements sont proposés, par exemple l'exposition « Shibori, l'indigo révélé » et les « Contes kamishibaï au jardin japonais » au Temps des Cerises.",
        "category": "location_multi",
        "source": "curated",
    },
    {
        "question": "Quels événements sont prévus à Courbevoie ?",
        "ground_truth": "À Courbevoie, on trouve notamment l'exposition « Écrire la ville » au Musée Roybet Fould (20 et 21 septembre 2025), la « Visite libre du Pavillon des Indes », ainsi que des ateliers emploi de France Travail.",
        "category": "location_multi",
        "source": "curated",
    },
    {
        "question": "Qu'y a-t-il à faire à Colombes ?",
        "ground_truth": "À Colombes, l'Avant-Seine - Théâtre de Colombes propose l'événement « TROC - Happy Culture ! » (5 septembre 2025) et une « Visite guidée des coulisses » (20 septembre 2025), en plus d'ateliers de recrutement.",
        "category": "location_multi",
        "source": "curated",
    },
    {
        "question": "Quels événements ont lieu à Nanterre ?",
        "ground_truth": "À Nanterre, on peut voir « Au Sommet - Cordée Circassienne » aux Arènes de Nanterre (20 et 21 novembre 2025), le spectacle « ASSEMBLÉE » Place de la cathédrale, et le concert « TM+ - Vanishings » à la Maison de la musique.",
        "category": "location_multi",
        "source": "curated",
    },
    {
        "question": "Que peut-on faire à Meudon ?",
        "ground_truth": "À Meudon, le Musée d'art et d'histoire propose une « Visite Flash », « La forêt s'éveille sur les coteaux » se tient Place Henri Brousse, et les « Dimanches au Vert en famille » ont lieu au Parc de l'Observatoire.",
        "category": "location_multi",
        "source": "curated",
    },
    {
        "question": "Quels événements à Boulogne-Billancourt ?",
        "ground_truth": "À Boulogne-Billancourt, le Musée Albert-Kahn propose le festival photographique « Mondes en commun » et une « Action collective », tandis que le Musée Paul Belmondo accueille « Lire les yeux fermés ».",
        "category": "location_multi",
        "source": "curated",
    },
    {
        "question": "Quels événements sont organisés à Saint-Cloud ?",
        "ground_truth": "À Saint-Cloud, le Musée des Avelines propose « Broderie royale », le Domaine national présente la maquette du pavillon Turc, et la scène ouverte « Open Zik » se tient au Carré (gratuit).",
        "category": "location_multi",
        "source": "curated",
    },
    # --- edge_unknown: subject absent from the corpus (must refuse) ---------------------
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
        "question": "Où voir un match de basket NBA dans les Hauts-de-Seine ?",
        "ground_truth": "Aucun match de basket NBA n'est proposé : je n'ai pas trouvé d'événement de ce type dans les données.",
        "category": "edge_unknown",
        "source": "curated",
    },
    {
        "question": "Y a-t-il une compétition de surf de prévue ?",
        "ground_truth": "Non, je n'ai pas trouvé de compétition de surf dans les données des Hauts-de-Seine.",
        "category": "edge_unknown",
        "source": "curated",
    },
    {
        "question": "Quand a lieu le festival de montgolfières ?",
        "ground_truth": "Je n'ai pas trouvé de festival de montgolfières dans les événements disponibles.",
        "category": "edge_unknown",
        "source": "curated",
    },
    {
        "question": "Y a-t-il une course de chiens de traîneau organisée ?",
        "ground_truth": "Non, aucune course de chiens de traîneau ne figure dans les données.",
        "category": "edge_unknown",
        "source": "curated",
    },
    {
        "question": "Où assister à une corrida dans les Hauts-de-Seine ?",
        "ground_truth": "Je n'ai pas trouvé d'événement de corrida dans les données.",
        "category": "edge_unknown",
        "source": "curated",
    },
    {
        "question": "Y a-t-il un safari animalier à proximité ?",
        "ground_truth": "Non, aucun safari animalier ne figure parmi les événements disponibles.",
        "category": "edge_unknown",
        "source": "curated",
    },
    # --- edge_outofscope: outside the 92 scope, or not an event question ----------------
    {
        "question": "Où puis-je assister à un Grand Prix de Formule 1 dans les Hauts-de-Seine ?",
        "ground_truth": "Aucun Grand Prix de Formule 1 n'est proposé : il n'y a pas d'événement de ce type dans les Hauts-de-Seine dans les données.",
        "category": "edge_outofscope",
        "source": "curated",
    },
    {
        "question": "Quels événements culturels y a-t-il à Lyon ?",
        "ground_truth": "Aucun événement à Lyon n'est disponible : les données ne couvrent que le département des Hauts-de-Seine (92).",
        "category": "edge_outofscope",
        "source": "curated",
    },
    {
        "question": "Que peut-on faire à Bordeaux ce week-end ?",
        "ground_truth": "Aucun événement à Bordeaux n'est disponible : les données ne couvrent que le département des Hauts-de-Seine (92).",
        "category": "edge_outofscope",
        "source": "curated",
    },
    {
        "question": "Quel temps fera-t-il ce week-end dans les Hauts-de-Seine ?",
        "ground_truth": "Je ne peux pas répondre à cette question : je renseigne uniquement sur les événements culturels des Hauts-de-Seine, pas sur la météo.",
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
        "--factual",
        type=int,
        default=60,
        help="Number of generated factual_lookup pairs (default: 60). Added to the ~40 "
        "curated cases for a ~100-pair, balanced test set.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.corpus, dtype=str, keep_default_na=False)
    usable = df[df.apply(_is_usable, axis=1)].drop_duplicates(subset="title")

    n_factual = max(args.factual, 0)
    if len(usable) < n_factual:
        raise SystemExit(
            f"Only {len(usable)} usable events for {n_factual} factual questions; "
            "lower --factual or grow the corpus (evaluation/build_corpus.py)."
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
