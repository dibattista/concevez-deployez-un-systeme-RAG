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


if __name__ == "__main__":
    unittest.main()
