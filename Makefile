# Lancement local du système RAG (voir README §1 pour le détail).
# Toutes les cibles passent par Poetry ; pré-requis : `make install` puis un .env
# contenant MISTRAL_API_KEY (cf. .env.example).

.DEFAULT_GOAL := help
.PHONY: help install collect clean index pipeline api ui query test eval \
	docker-build docker-up docker-down docker-logs

help: ## Affiche les cibles disponibles
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Installe les dépendances (Poetry)
	poetry install

collect: ## Étape 1 — collecte des événements OpenAgenda -> data/openagenda_idf_events.csv
	poetry run python scripts/collect_events.py

clean: ## Étape 2 — nettoyage -> data/events_clean.csv
	poetry run python scripts/clean_events.py

index: ## Étape 3 — construction de l'index FAISS -> faiss_index/
	poetry run python scripts/build_vector_store.py

pipeline: collect clean index ## Pipeline complet de données (collecte -> nettoyage -> index)

api: ## Lance l'API REST FastAPI (Swagger sur /docs) -> http://localhost:8000
	poetry run uvicorn src.api.main:app

ui: ## Lance l'interface de chat Dash -> http://localhost:8050
	poetry run python interface/dash_app.py

query: ## Interroge le RAG en CLI : make query q="Quels concerts à Paris ?"
	poetry run python scripts/rag_query.py "$(q)"

test: ## Lance les tests unitaires (hors ligne, sans clé API)
	poetry run pytest

eval: ## Lance l'évaluation Ragas (nécessite MISTRAL_API_KEY)
	poetry run python evaluate_rag.py

docker-build: ## Construit l'image Docker (API + UI)
	docker compose build

docker-up: ## Lance la stack conteneurisée -> API http://localhost:8000/docs · UI http://localhost:8050
	docker compose up -d --build
	@echo "API  -> http://localhost:8000/docs"
	@echo "UI   -> http://localhost:8050"

docker-down: ## Arrête la stack conteneurisée (l'index reste sur l'hôte)
	docker compose down

docker-logs: ## Suit les logs des conteneurs
	docker compose logs -f
