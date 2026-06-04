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

### Embedding

### Modèle utilisé

### Dimensionnalité, logique de batch, format des vecteurs

## 4. Choix du modèle NLP

### Modèle sélectionné

### Pourquoi ce modèle ?

### Prompting (si utilisé)

### Limites du modèle

## 5. Construction de la base vectorielle

### Faiss utilisé

### Stratégie de persistance

### Format de sauvegarde

### Nommage

### Métadonnées associées

### Ce qui est conservé pour chaque document

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
