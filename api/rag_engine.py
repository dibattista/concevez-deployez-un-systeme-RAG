import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage


class RAGEngine:
    def __init__(self, faiss_index_path="data/faiss_index"):
        # 1. Chargement des variables d'environnement
        load_dotenv()

        # Vérification de la clé API Mistral
        if not os.getenv("MISTRAL_API_KEY"):
            raise ValueError(
                "La clé API Mistral (MISTRAL_API_KEY) "
                "n'est pas définie dans l'environnement."
            )

        # 2. Vérification de l'existence de l'index FAISS
        if not os.path.exists(faiss_index_path):
            raise FileNotFoundError(
                "L'index FAISS n'existe pas à l'emplacement : "
                f"'{faiss_index_path}'"  # noqa: E501
            )

        # 3. Chargement de l'embedding et de l'index
        try:
            self.embeddings = MistralAIEmbeddings(model="mistral-embed")
        except Exception as e:
            raise RuntimeError(
                "Erreur lors du chargement de l'embedding."
            ) from e  # noqa: E501

        try:
            self.db = FAISS.load_local(
                faiss_index_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            raise RuntimeError(
                "Erreur lors du chargement de l'index FAISS."
            ) from e  # noqa: E501

        # Initialisation du LLM
        self.llm = ChatMistralAI(model="mistral-large-latest", temperature=0)

    def ask(self, question: str, k: int = 3) -> dict:
        # 1. Recherche des documents pertinents
        try:
            docs_and_scores = self.db.similarity_search_with_score(
                question, k=k
            )  # noqa: E501
        except Exception as e:
            raise ValueError(
                f"Erreur lors de la recherche sémantique: {e}"
            ) from e  # noqa: E501

        if not docs_and_scores:
            raise ValueError(
                f"Aucun document trouvé pour la question : '{question}'"
            )  # noqa: E501

        # 2. Construction du contexte + des sources (uid + score)
        contexts = []
        sources = []
        for doc, score in docs_and_scores:
            contexts.append(doc.page_content)
            sources.append(
                {"uid": doc.metadata.get("uid"), "score": float(score)}
            )  # noqa: E501

        context_text = "\n\n".join(contexts)

        # 3. Construction du prompt
        system_prompt = (
            "Tu es un assistant spécialisé dans les événements et le RAG.\n"
            "Réponds à la question de l'utilisateur de manière précise, "
            "concise et en français, en utilisant uniquement le contexte "
            "fourni ci-dessous. Si le contexte ne contient pas les "
            "informations nécessaires pour répondre, dis que tu ne sais pas."
        )
        user_prompt = f"Contexte:\n{context_text}\n\nQuestion: {question}"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # 4. Génération de la réponse
        try:
            response = self.llm.invoke(messages)
        except Exception as e:
            raise RuntimeError(
                f"Erreur lors de la génération de la réponse: {e}"
            ) from e

        return {
            "answer": response.content,
            "sources": sources,
        }
