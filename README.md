# Concevez et déployez un système RAG

Projet de conception et déploiement d'un système RAG (Retrieval-Augmented Generation) utilisant LangChain, FAISS et Mistral AI, exposé via une API FastAPI.

Ce système récupère les événements culturels et artistiques en région Auvergne-Rhône-Alpes depuis l'API OpenAgenda, filtre et nettoie les données, génère des représentations vectorielles (embeddings sémantiques) avec Mistral AI, puis les stocke dans un index local FAISS. Une classe de moteur RAG (`RAGEngine`) et des tests unitaires complets sont fournis pour interroger le système et valider son fonctionnement.

---

## Structure du projet

- `api/` : Code du module API RAG, incluant la classe `RAGEngine`.
- `data/` : Dossier contenant les données brutes, nettoyées, et l'index FAISS local.
- `scripts/` : Scripts utilitaires pour l'ingestion, le nettoyage, la génération d'index et le test en ligne de commande.
- `tests/` : Suite complète de tests unitaires (ingestion, moteur RAG).
- `venv/` : Environnement virtuel Python.

---

## Installation et Configuration

### 1. Cloner ou initialiser le dépôt

### 2. Créer et activer l'environnement virtuel
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Configurer les variables d'environnement
Copiez le fichier de configuration d'exemple :
```bash
cp .env.example .env
```
Remplissez ensuite votre fichier `.env` avec vos clés d'API :
- `MISTRAL_API_KEY` : Clé d'API Mistral AI (nécessaire pour l'embedding et le chat LLM).
- `OPENAGENDA_API_KEY` : Clé d'API OpenAgenda (nécessaire pour récupérer les événements).

---

## Utilisation du Pipeline RAG

### Étape 1 : Récupération et nettoyage des événements
Ce script interroge l'API OpenAgenda pour obtenir les événements de la région Auvergne-Rhône-Alpes, effectue une déduplication, filtre les événements sans localisation, extrait uniquement les textes en français, et sauvegarde les données nettoyées.
```bash
python scripts/fetch_events.py
```
*Fichiers générés :*
- `data/events_raw.json` (événements bruts)
- `data/events_clean.json` (événements nettoyés et filtrés)

### Étape 2 : Construction de l'index de recherche FAISS
Ce script prépare des chunks textuels à partir de `data/events_clean.json`, génère leurs représentations vectorielles via le modèle `mistral-embed` et construit l'index FAISS local.
```bash
python scripts/build_index.py
```
*Dossier généré :*
- `data/faiss_index/` (fichiers d'index FAISS stockés localement)

### Étape 3 : Tester le système RAG en ligne de commande
Vous pouvez utiliser le script de test pour effectuer une recherche sémantique et générer une réponse de manière interactive :
```bash
python scripts/test_rag.py "Quels évènements concernent la facturation électronique ?"
```
Options disponibles :
- `--k <int>` : Nombre de documents pertinents à récupérer (défaut: 3).
- `--model <str>` : Modèle Mistral à utiliser (défaut: `mistral-large-latest`).
- `--no-gen` : Désactive la génération par LLM et retourne uniquement les documents trouvés.

---

## Utilisation de la classe `RAGEngine`

La classe `RAGEngine` (située dans `api/rag_engine.py`) encapsule l'intégralité du moteur RAG :
```python
from api import RAGEngine

# Initialisation du moteur (charge l'index FAISS et configure Mistral AI)
engine = RAGEngine(faiss_index_path="data/faiss_index")

# Poser une question au système
result = engine.ask("Quels évènements concernent la facturation électronique ?", k=3)

print("Réponse :", result["answer"])
print("Sources :", result["sources"])
```

---

## Tests Unitaires

Une suite complète de tests unitaires est disponible pour valider la récupération, le nettoyage et le moteur RAG. Pour lancer tous les tests de façon automatique, exécutez la commande suivante :
```bash
python -m unittest discover tests
```
