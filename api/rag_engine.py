import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI


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
