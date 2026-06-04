# Rapport technique – Assistant intelligent de recommandation d’événements culturels

> POC d'un système de génération augmentée par récupération (RAG) répondant en
> langage naturel à des questions sur les évênements culturels à venir.
> Mission réalisée pour **Puls-Events**, plateforme de recommendations culturelles
> personnalisées, à destination de ses équipes produits et marketing.

--

## 1. Objectifs du projet

### Contexte

**Puls-Events** développe une plateforme de **recommandations culturelles
personnalisées**. L'entreprise souhaité évaluer la faisabilité de l'intégration
d'un **chatbot intelligent** capable de répondre aux questions des utilisateurs sur
des évênements culturels, en s'appuyant sur les données ouvertes de l'API
**OpenAgenda**.

Ces données sont riches mais difficilement exploitables par un utilisateur
en raison de leur nature non-structurée. La recherche par mots-clés classique
ne peut pas capturer l'intention réelle d'une question formulée en langage naturel
("Quels concerts de musique classique se passent à Paris ce week-end ?"). Un
système RAG permet de combler ces limites en combinant recherche sémantqiue et
génération de réponse pour offrir une expérience conversationnelle proche
d'une recommendation personnalisée.

### Problématique

### Objectif du POC

### Périmètre

## 2. Architecture du système

### Schéma global (schéma UML)

### Données entrantes (API Open Agenda)

Les données proviennent du jeu de données ouvert **« Événements publics en
Île-de-France (via Open Agenda) »** (`evenements-publics-cibul`), exposé par
l'**API Explore v2.1 d'Opendatasoft** (`data.iledefrance.fr`) — aucune clé API
requise. Le pipeline de données suit deux étapes séparées et testées :

1. **Collecte** (`scripts/collect_events.py` → `src/data/fetch_openagenda.py`) :
   export brut des événements vers `data/openagenda_idf_events.{json,csv}`.
2. **Nettoyage** (`scripts/clean_events.py` → `src/data/clean.py`) : structuration
   et fiabilisation vers `data/events_clean.csv`, qui alimente l'indexation FAISS.

### Prétraitement / embeddings / base vectorielle

### Intégration LLM avec LangChain

### Exposition via API

### Technologies utilisées

## 3. Préparation et vectorisation des données

### Source de données

**API** : Opendatasoft Explore v2.1, dataset `evenements-publics-cibul`
(OpenAgenda, région Île-de-France). La récupération utilise l'endpoint d'**export
en masse** (`/exports/json`), qui renvoie l'intégralité du résultat filtré en une
requête et contourne ainsi la limite de 10 000 enregistrements de l'endpoint
paginé `/records`.

**Filtre de collecte** : un seul filtre temporel,
`lastdate_end >= aujourd'hui - 365 jours`. Un événement est conservé si sa date de
fin est postérieure à cette borne, ce qui capture **la dernière année** ainsi que
**tous les événements à venir** (en cours et programmés).

**Champs récupérés** : un sous-ensemble de 25 champs pertinents pour le RAG
(identifiants et URL, titres et descriptions FR, dates, localisation, conditions,
mots-clés…). Volume brut obtenu : **~20 552 événements**.

```bash
# Collecte (fenêtre par défaut : 365 jours)
poetry run python scripts/collect_events.py
poetry run python scripts/collect_events.py --days 365 --all-fields
```

### Nettoyage

Le jeu brut est bruité (descriptions en HTML, dates non parsées, codes postaux lus
comme des flottants, noms de départements en texte libre avec de nombreuses
variantes, doublons). `src/data/clean.py` le transforme en un jeu **structuré et
fiable** (`data/events_clean.csv`) via les étapes suivantes :

| Étape | Traitement |
|---|---|
| Texte | Suppression des balises HTML (BeautifulSoup) + décodage des entités, normalisation des espaces |
| Événements vides | Suppression des lignes sans aucun texte exploitable (ni description ni titre) |
| Dates | Parsing de `firstdate_begin` / `lastdate_end` en `datetime` UTC (`NaT` si absente/invalide) |
| Code postal | Normalisation en chaîne de 5 caractères (`75014.0` → `"75014"`) |
| Département | Code à 2 chiffres dérivé du code postal (fiable), avec repli sur le nom du département pour les variantes orthographiques |
| Mots-clés | Aplatissement de la liste sérialisée `"['Jazz', 'Concert']"` → `"Jazz, Concert"` |
| Périmètre IDF | Conservation des seuls départements franciliens (75, 77, 78, 91, 92, 93, 94, 95) |
| Doublons | Déduplication sur l'`id` de l'événement (la mise à jour la plus récente l'emporte) |

**Choix de schéma** : seuls les champs **fiablement dérivables** sont conservés.
Les colonnes `keywords` et `conditions` sont gardées comme contexte brut plutôt que
de fabriquer des champs `category` / `price_type` à partir de texte libre bruité.
Schéma de sortie : `id`, `title`, `description`, `summary`, `date_start`,
`date_end`, `daterange`, `venue`, `address`, `city`, `postalcode`, `department`,
`coordinates`, `keywords`, `conditions`, `url`, `updatedat`.

**Résultat** : **20 177 / 20 552** événements conservés (≈ 375 supprimés : hors
Île-de-France ou sans texte exploitable).

```bash
# Nettoyage : data/openagenda_idf_events.csv -> data/events_clean.csv
poetry run python scripts/clean_events.py
```

Les transformations sont couvertes par des tests unitaires
(`tests/test_fetch.py`, `tests/test_clean.py`) : `poetry run pytest`.

### Chunking

Chaque événement est d'abord transformé en `Document` LangChain dont le contenu est
**préfixé d'un en-tête de métadonnées** (titre / date / lieu) avant vectorisation :

```
Titre: {title}
Date: {daterange}
Lieu: {venue}, {city} ({postalcode})

{description}
```

Cet ancrage permet à la recherche sémantique de matcher sur le *quoi / quand / où*
même lorsque la description est succincte. Le découpage utilise
`RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)`. La fenêtre est
volontairement large : les descriptions d'événements sont courtes (souvent un seul
chunk), ce qui **préserve la cohérence de l'événement** tout en respectant le prérequis
de chunking. Sur le corpus nettoyé, ~20 177 événements produisent un nombre de chunks
légèrement supérieur (peu d'événements dépassent 2000 caractères).

### Embedding

### Modèle utilisé

**`mistral-embed`** (via `MistralAIEmbeddings`) — choix arrêté. Le cahier des charges
oriente vers Mistral pour la vectorisation, ce qui garantit un stack cohérent (même
provider/API que le LLM de génération) et de bonnes performances en français.

### Dimensionnalité, logique de batch, format des vecteurs

- **Dimensionnalité** : vecteurs de **1024** dimensions.
- **Batch** : les chunks sont embarqués par lots (`--batch-size`, défaut 64) pour
  respecter les limites de débit de l'API Mistral ; progression affichée
  (`indexed N/total`).
- **Format** : index FAISS `IndexFlatL2` (distance L2), adapté au volume modeste du
  corpus ; pas besoin d'index approximatif (IVF/HNSW) à cette échelle.

```bash
# Indexation : data/events_clean.csv -> faiss_index/
poetry run python scripts/build_vector_store.py
poetry run python scripts/build_vector_store.py --limit 200   # run de test rapide
```

## 4. Choix du modèle NLP

### Modèle sélectionné

### Pourquoi ce modèle ?

### Prompting (si utilisé)

### Limites du modèle

## 5. Construction de la base vectorielle

### Faiss utilisé

Base vectorielle **FAISS** (imposée). Index **`IndexFlatL2`** (recherche exhaustive,
distance L2) : à l'échelle du corpus (~20 k événements), un index *flat* offre une
précision exacte sans coût notable. Un passage à un index approximatif (IVF, HNSW) ne
serait pertinent qu'en cas de montée en charge importante.

### Stratégie de persistance

L'index est **construit une fois puis persisté sur disque** (`faiss_index/`), de sorte
que le composant de retrieval le recharge sans re-vectoriser. La reconstruction est
entièrement reproductible à partir de `data/events_clean.csv` via
`scripts/build_vector_store.py`. Le dossier `faiss_index/` est *gitignoré* (artefact
régénérable, non versionné).

### Format de sauvegarde

`FAISS.save_local()` écrit deux fichiers :

- `index.faiss` : les vecteurs et la structure de l'index FAISS.
- `index.pkl` : le *docstore* (contenu des `Document` + métadonnées) et la table de
  correspondance `id ↔ vecteur`.

Le rechargement se fait via `load_vector_store()`
(`FAISS.load_local(..., allow_dangerous_deserialization=True)` — sûr ici car l'index
est produit par le projet lui-même).

### Métadonnées associées

### Ce qui est conservé pour chaque document

Pour chaque chunk indexé, on stocke aux côtés du vecteur les métadonnées utiles à la
**citation des sources** et au **pré-filtrage** par la chaîne RAG :

| Champ | Usage |
|---|---|
| `id` | Identifiant de l'événement (déduplication / référence) |
| `title` | Titre, cité dans la réponse |
| `daterange`, `date_start`, `date_end` | Dates lisibles + bornes pour le filtrage temporel |
| `venue`, `city`, `postalcode`, `department` | Lieu, cité et utilisé pour le filtrage géographique |
| `keywords` | Mots-clés, contexte additionnel |
| `url` | Lien canonique vers la fiche de l'événement |

Le **contenu vectorisé** (`page_content`) reste l'en-tête de métadonnées suivi de la
description nettoyée (cf. § Chunking).

## 6. API et endpoints exposés

### Framework utilisé

### Endpoints clés

### Format des requêtes/réponses

### Exemple d’appel API

### Tests effectués et documentés

### Gestion des erreurs / limitations

## 7. Évaluation du système

### Jeu de test annoté

### Nombre d’exemples

### Méthode d’annotation

### Métriques d’évaluation

### Résultats obtenus

### Analyse quantitative

### Analyse qualitative

## 8. Recommandations et perspectives

### Ce qui fonctionne bien

### Limites du POC

### Améliorations possibles

### Passage en production via…

## 9. Organisation du dépôt GitHub

### Arborescence du dépôt

### Explication rapide de chaque répertoire

## 10. Annexes (exemples)

### Extraits du jeu de test annoté

### Prompt utilisé

### Extraits de logs ou exemples de réponse JSON
