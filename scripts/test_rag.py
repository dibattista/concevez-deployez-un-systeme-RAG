import os
import sys
import argparse
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage


def main():
    # Setup CLI arguments
    parser = argparse.ArgumentParser(
        description=(
            "Test RAG: Recherche sémantique et génération "
            "avec Mistral AI & FAISS."
        )  # fmt: skip
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default="Quels évènements concernent la facturation électronique ?",
        help=(
            "La question à poser au système RAG "
            "(défaut: 'Quels évènements concernent la facturation "
            "électronique ?')"
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Nombre de documents à récupérer (défaut: 3)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistral-large-latest",
        help=(
            "Modèle de chat Mistral à utiliser "
            "(défaut: mistral-large-latest)"
        ),  # fmt: skip
    )
    parser.add_argument(
        "--no-gen",
        action="store_true",
        help=(
            "Désactive la génération de réponse par LLM "
            "(effectue uniquement la recherche)"
        ),
    )
    args = parser.parse_args()

    # 1. Charger l'environnement
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("Erreur: MISTRAL_API_KEY n'est pas définie dans le fichier .env")
        sys.exit(1)

    index_dir = os.path.join("data", "faiss_index")
    if not os.path.exists(index_dir):
        print(
            "Erreur: Index FAISS introuvable à l'emplacement "
            f"'{index_dir}'."
        )  # fmt: skip
        print(
            "Veuillez d'abord exécuter le script d'indexation: "
            "python scripts/build_index.py"
        )
        sys.exit(1)

    print("--- RAG TEST SCRIPT ---")
    print(f"Question : '{args.query}'\n")

    # 2. Initialiser les embeddings et charger l'index FAISS
    print(
        "Chargement de l'index FAISS et "
        "initialisation du modèle d'embedding..."
    )  # fmt: skip
    try:
        embeddings = MistralAIEmbeddings(model="mistral-embed")
        db = FAISS.load_local(
            index_dir, embeddings, allow_dangerous_deserialization=True
        )
        print("Index FAISS chargé avec succès.\n")
    except Exception as e:
        print(f"Erreur lors du chargement de l'index: {e}")
        sys.exit(1)

    # 3. Effectuer la recherche sémantique
    print(f"Recherche des {args.k} documents les plus pertinents...")
    try:
        docs_and_scores = db.similarity_search_with_score(args.query, k=args.k)
    except Exception as e:
        print(f"Erreur lors de la recherche sémantique: {e}")
        sys.exit(1)

    if not docs_and_scores:
        print("Aucun document trouvé.")
        sys.exit(0)

    # Afficher les documents récupérés
    print("\n=== DOCUMENTS RETROUVÉS (CONTEXTE) ===")
    contexts = []
    for i, (doc, score) in enumerate(docs_and_scores, 1):
        print(
            f"\n[Document {i}] - Score L2 (plus bas est meilleur): "
            f"{score:.4f}"
        )  # fmt: skip
        print(f"Métadonnées : {doc.metadata}")
        print(f"Contenu :\n{doc.page_content}")
        print("-" * 50)
        contexts.append(doc.page_content)

    if args.no_gen:
        print("\nMode sans génération activé. Fin du script.")
        sys.exit(0)

    # 4. Générer la réponse avec le LLM
    print(
        f"\nInitialisation de la génération avec le "
        f"modèle '{args.model}'..."
    )  # fmt: skip
    context_text = "\n\n".join(contexts)

    system_prompt = (
        "Tu es un assistant spécialisé dans les événements et le RAG.\n"
        "Réponds à la question de l'utilisateur de manière précise, "
        "concise et en français, en utilisant uniquement le contexte "
        "fourni ci-dessous. Si le contexte ne contient pas les "
        "informations nécessaires pour répondre, dis que tu ne sais pas."
    )

    user_prompt = f"Contexte:\n{context_text}\n\nQuestion: {args.query}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    try:
        llm = ChatMistralAI(model=args.model, temperature=0)
        print("Génération de la réponse en cours...")
        response = llm.invoke(messages)
        print("\n=== RÉPONSE DE MISTRAL ===")
        print(response.content)
        print("==========================")
    except Exception as e:
        print(
            f"\nErreur lors de la génération avec le modèle "
            f"'{args.model}': {e}"
        )  # fmt: skip
        print("Tentative de repli vers 'open-mixtral-8x7b'...")
        try:
            llm_fallback = ChatMistralAI(
                model="open-mixtral-8x7b", temperature=0
            )  # fmt: skip
            response = llm_fallback.invoke(messages)
            print("\n=== RÉPONSE DE MISTRAL (Modèle de repli) ===")
            print(response.content)
            print("==========================")
        except Exception as fallback_err:
            print(f"Erreur de repli également: {fallback_err}")
            print(
                "Impossible de générer une réponse. Les résultats de "
                "recherche ci-dessus restent valides."
            )


if __name__ == "__main__":
    main()
