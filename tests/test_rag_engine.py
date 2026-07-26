import unittest
from unittest.mock import patch, MagicMock
from api.rag_engine import RAGEngine


class TestRAGEngine(unittest.TestCase):
    """
    Tests unitaires pour la classe RAGEngine.
    """

    @patch("api.rag_engine.load_dotenv")
    @patch("api.rag_engine.os.getenv")
    def test_missing_api_key_raises_value_error(
        self, mock_getenv, mock_load_dotenv
    ):  # noqa: E501
        """
        Vérifie qu'une ValueError est levée si MISTRAL_API_KEY
        n'est pas définie.
        """
        mock_getenv.return_value = None

        with self.assertRaises(ValueError) as context:
            RAGEngine("data/faiss_index")

        self.assertIn("MISTRAL_API_KEY", str(context.exception))

    @patch("api.rag_engine.load_dotenv")
    @patch("api.rag_engine.os.getenv")
    @patch("api.rag_engine.os.path.exists")
    def test_missing_index_raises_file_not_found_error(
        self, mock_exists, mock_getenv, mock_load_dotenv
    ):
        """
        Vérifie qu'une FileNotFoundError est levée si l'index FAISS
        n'existe pas.
        """
        mock_getenv.return_value = "mock_api_key"
        mock_exists.return_value = False

        with self.assertRaises(FileNotFoundError) as context:
            RAGEngine("nonexistent_index_path")

        self.assertIn("L'index FAISS n'existe pas", str(context.exception))

    @patch("api.rag_engine.load_dotenv")
    @patch("api.rag_engine.os.getenv")
    @patch("api.rag_engine.os.path.exists")
    @patch("api.rag_engine.MistralAIEmbeddings")
    def test_embeddings_load_error_raises_runtime_error(
        self, mock_embeddings, mock_exists, mock_getenv, mock_load_dotenv
    ):
        """
        Vérifie qu'une RuntimeError est levée en cas d'erreur de
        chargement de l'embedding (avec from e).
        """
        mock_getenv.return_value = "mock_api_key"
        mock_exists.return_value = True

        # Simuler une exception lors de la création de MistralAIEmbeddings
        underlying_exception = Exception(
            "Connection error to Mistral Embeddings API"
        )  # noqa: E501
        mock_embeddings.side_effect = underlying_exception

        with self.assertRaises(RuntimeError) as context:
            RAGEngine("data/faiss_index")

        self.assertIn(
            "Erreur lors du chargement de l'embedding", str(context.exception)
        )
        # Vérifier que "from e" (cause) est correctement propagé
        self.assertEqual(context.exception.__cause__, underlying_exception)

    @patch("api.rag_engine.load_dotenv")
    @patch("api.rag_engine.os.getenv")
    @patch("api.rag_engine.os.path.exists")
    @patch("api.rag_engine.MistralAIEmbeddings")
    @patch("api.rag_engine.FAISS.load_local")
    def test_faiss_load_error_raises_runtime_error(
        self,
        mock_faiss,
        mock_embeddings,
        mock_exists,
        mock_getenv,
        mock_load_dotenv,
    ):
        """
        Vérifie qu'une RuntimeError est levée en cas d'erreur de
        chargement de l'index FAISS.
        """
        mock_getenv.return_value = "mock_api_key"
        mock_exists.return_value = True
        mock_embeddings.return_value = MagicMock()

        # Simuler une exception lors du chargement de l'index FAISS
        underlying_exception = Exception("FAISS index deserialization failed")
        mock_faiss.side_effect = underlying_exception

        with self.assertRaises(RuntimeError) as context:
            RAGEngine("data/faiss_index")

        self.assertIn(
            "Erreur lors du chargement de l'index FAISS",
            str(context.exception),
        )
        # Vérifier que "from e" (cause) est correctement propagé
        self.assertEqual(context.exception.__cause__, underlying_exception)

    @patch("api.rag_engine.load_dotenv")
    @patch("api.rag_engine.os.getenv")
    @patch("api.rag_engine.os.path.exists")
    @patch("api.rag_engine.MistralAIEmbeddings")
    @patch("api.rag_engine.FAISS.load_local")
    @patch("api.rag_engine.ChatMistralAI")
    def test_successful_initialization(
        self,
        mock_chat,
        mock_faiss,
        mock_embeddings,
        mock_exists,
        mock_getenv,
        mock_load_dotenv,
    ):
        """
        Vérifie que l'initialisation réussit et définit
        correctement les attributs self.embeddings, self.db,
        et self.llm.
        """
        mock_getenv.return_value = "mock_api_key"
        mock_exists.return_value = True

        mock_embed_instance = MagicMock()
        mock_embeddings.return_value = mock_embed_instance

        mock_db_instance = MagicMock()
        mock_faiss.return_value = mock_db_instance

        mock_llm_instance = MagicMock()
        mock_chat.return_value = mock_llm_instance

        engine = RAGEngine("data/faiss_index")

        # Vérifier les attributs de l'instance
        self.assertEqual(engine.embeddings, mock_embed_instance)
        self.assertEqual(engine.db, mock_db_instance)
        self.assertEqual(engine.llm, mock_llm_instance)

        # Vérifier les appels d'initialisation
        mock_embeddings.assert_called_once_with(model="mistral-embed")
        mock_faiss.assert_called_once_with(
            "data/faiss_index",
            mock_embed_instance,
            allow_dangerous_deserialization=True,
        )
        mock_chat.assert_called_once_with(
            model="mistral-large-latest", temperature=0
        )  # noqa: E501

    @patch("api.rag_engine.load_dotenv")
    @patch("api.rag_engine.os.getenv")
    @patch("api.rag_engine.os.path.exists")
    @patch("api.rag_engine.MistralAIEmbeddings")
    @patch("api.rag_engine.FAISS.load_local")
    @patch("api.rag_engine.ChatMistralAI")
    def test_ask_success(
        self,
        mock_chat,
        mock_faiss,
        mock_embeddings,
        mock_exists,
        mock_getenv,
        mock_load_dotenv,
    ):
        """
        Vérifie que ask() retourne la réponse attendue et les
        sources associées.
        """
        mock_getenv.return_value = "mock_api_key"
        mock_exists.return_value = True

        mock_db = MagicMock()
        mock_faiss.return_value = mock_db

        mock_doc1 = MagicMock()
        mock_doc1.page_content = "Contenu de test 1"
        mock_doc1.metadata = {"uid": "123"}

        mock_db.similarity_search_with_score.return_value = [(mock_doc1, 0.45)]

        mock_llm = MagicMock()
        mock_chat.return_value = mock_llm

        mock_response = MagicMock()
        mock_response.content = "Réponse générée"
        mock_llm.invoke.return_value = mock_response

        engine = RAGEngine()
        res = engine.ask("Ma question", k=2)

        self.assertEqual(res["answer"], "Réponse générée")
        self.assertEqual(res["sources"], [{"uid": "123", "score": 0.45}])
        mock_db.similarity_search_with_score.assert_called_once_with(
            "Ma question", k=2
        )  # noqa: E501

    @patch("api.rag_engine.load_dotenv")
    @patch("api.rag_engine.os.getenv")
    @patch("api.rag_engine.os.path.exists")
    @patch("api.rag_engine.MistralAIEmbeddings")
    @patch("api.rag_engine.FAISS.load_local")
    @patch("api.rag_engine.ChatMistralAI")
    def test_ask_search_error(
        self,
        mock_chat,
        mock_faiss,
        mock_embeddings,
        mock_exists,
        mock_getenv,
        mock_load_dotenv,
    ):
        """
        Vérifie que ask() lève ValueError en cas d'erreur de recherche.
        """
        mock_getenv.return_value = "mock_api_key"
        mock_exists.return_value = True

        mock_db = MagicMock()
        mock_faiss.return_value = mock_db
        mock_db.similarity_search_with_score.side_effect = Exception(
            "Search failed"
        )  # noqa: E501

        engine = RAGEngine()
        with self.assertRaises(ValueError) as context:
            engine.ask("Ma question")

        self.assertIn(
            "Erreur lors de la recherche sémantique", str(context.exception)
        )  # noqa: E501

    @patch("api.rag_engine.load_dotenv")
    @patch("api.rag_engine.os.getenv")
    @patch("api.rag_engine.os.path.exists")
    @patch("api.rag_engine.MistralAIEmbeddings")
    @patch("api.rag_engine.FAISS.load_local")
    @patch("api.rag_engine.ChatMistralAI")
    def test_ask_no_docs_found(
        self,
        mock_chat,
        mock_faiss,
        mock_embeddings,
        mock_exists,
        mock_getenv,
        mock_load_dotenv,
    ):
        """
        Vérifie que ask() lève ValueError si aucun document n'est trouvé.
        """
        mock_getenv.return_value = "mock_api_key"
        mock_exists.return_value = True

        mock_db = MagicMock()
        mock_faiss.return_value = mock_db
        mock_db.similarity_search_with_score.return_value = []

        engine = RAGEngine()
        with self.assertRaises(ValueError) as context:
            engine.ask("Ma question")

        self.assertIn(
            "Aucun document trouvé pour la question", str(context.exception)
        )  # noqa: E501

    @patch("api.rag_engine.load_dotenv")
    @patch("api.rag_engine.os.getenv")
    @patch("api.rag_engine.os.path.exists")
    @patch("api.rag_engine.MistralAIEmbeddings")
    @patch("api.rag_engine.FAISS.load_local")
    @patch("api.rag_engine.ChatMistralAI")
    def test_ask_generation_error(
        self,
        mock_chat,
        mock_faiss,
        mock_embeddings,
        mock_exists,
        mock_getenv,
        mock_load_dotenv,
    ):
        """
        Vérifie que ask() lève RuntimeError en cas d'erreur de génération.
        """
        mock_getenv.return_value = "mock_api_key"
        mock_exists.return_value = True

        mock_db = MagicMock()
        mock_faiss.return_value = mock_db

        mock_doc1 = MagicMock()
        mock_doc1.page_content = "Contenu de test 1"
        mock_doc1.metadata = {"uid": "123"}
        mock_db.similarity_search_with_score.return_value = [(mock_doc1, 0.45)]

        mock_llm = MagicMock()
        mock_chat.return_value = mock_llm
        mock_llm.invoke.side_effect = Exception("LLM crash")

        engine = RAGEngine()
        with self.assertRaises(RuntimeError) as context:
            engine.ask("Ma question")

        self.assertIn(
            "Erreur lors de la génération de la réponse",
            str(context.exception),  # noqa: E501
        )


if __name__ == "__main__":
    unittest.main()
