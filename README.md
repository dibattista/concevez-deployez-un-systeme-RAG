# Concevez et déployez un système RAG

Projet de conception et déploiement d'un système RAG (Retrieval-Augmented Generation) utilisant LangChain, FAISS et Mistral AI, exposé via une API FastAPI.

## Structure du projet

- `data/` : Dossier contenant les données (documents à indexer).
- `scripts/` : Scripts utilitaires pour l'ingestion des données et l'indexation.
- `api/` : Code de l'API FastAPI pour interroger le système RAG.
- `tests/` : Tests unitaires et d'intégration.
- `venv/` : Environnement virtuel Python.

## Installation

1. Cloner ou initialiser le dépôt.
2. Créer l'environnement virtuel : `python3 -m venv venv`
3. Activer l'environnement : `source venv/bin/activate`
4. Installer les dépendances : `pip install -r requirements.txt`
5. Configurer les variables d'environnement en copiant `.env.example` vers `.env` et en y insérant votre clé API Mistral.
