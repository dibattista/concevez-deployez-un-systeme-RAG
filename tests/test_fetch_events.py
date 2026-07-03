import os
import json
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

# Définir une clé API de test avant d'importer le script
# Cela évite que le script ne s'arrête avec sys.exit(1) s'il ne trouve pas la clé
os.environ["OPENAGENDA_API_KEY"] = "mock_openagenda_api_key_for_testing"

from scripts.fetch_events import (
    clean_and_save_events,
    fetch_all_agendas,
    fetch_events_for_agenda,
)


class TestFetchEvents(unittest.TestCase):
    """
    Tests unitaires pour valider les fonctionnalités de scripts/fetch_events.py
    """

    @patch("scripts.fetch_events.requests.get")
    def test_fetch_all_agendas_success_and_pagination(self, mock_get):
        """
        Teste la récupération des agendas avec pagination (curseur 'after').
        """
        # Configuration des réponses mockées pour requests.get
        # Page 1 : contient un agenda et un curseur pour la page suivante
        response_page_1 = MagicMock()
        response_page_1.status_code = 200
        response_page_1.json.return_value = {
            "agendas": [{"uid": 123, "title": "Agenda Auvergne 1"}],
            "after": "cursor_next_page",
        }

        # Page 2 : contient un deuxième agenda et aucun curseur 'after' (fin)
        response_page_2 = MagicMock()
        response_page_2.status_code = 200
        response_page_2.json.return_value = {
            "agendas": [{"uid": 456, "title": "Agenda Auvergne 2"}],
            "after": None,
        }

        # Définition de l'ordre des réponses du mock requests.get
        mock_get.side_effect = [response_page_1, response_page_2]

        # Exécution de la fonction
        agendas = fetch_all_agendas("auvergne-rhone-alpes")

        print("\n✅ Connexion à l'API + clé valide (testé avec mock)")
        print("✅ Récupération de tous les agendas et pagination vérifiée")
        print("✅ Filtre par région vérifié")

        # Vérifications
        self.assertEqual(len(agendas), 2)
        self.assertEqual(agendas[0]["uid"], 123)
        self.assertEqual(agendas[1]["uid"], 456)

        # Vérifier que requests.get a été appelé 2 fois
        self.assertEqual(mock_get.call_count, 2)

        # Vérifier les paramètres passés lors des appels
        # Premier appel : sans curseur
        first_call_args = mock_get.call_args_list[0]
        self.assertIn("search", first_call_args[1]["params"])
        self.assertEqual(
            first_call_args[1]["params"]["search"], "auvergne-rhone-alpes"
        )
        self.assertNotIn("after[]", first_call_args[1]["params"])

        # Deuxième appel : avec curseur 'after[]'
        second_call_args = mock_get.call_args_list[1]
        self.assertIn("after[]", second_call_args[1]["params"])
        self.assertEqual(
            second_call_args[1]["params"]["after[]"], "cursor_next_page"
        )

    @patch("scripts.fetch_events.requests.get")
    def test_fetch_events_for_agenda_params(self, mock_get):
        """
        Teste la récupération des événements avec les filtres de date.
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "events": [{"uid": 789, "title": "Super Événement"}],
            "after": None,
        }
        mock_get.return_value = mock_response

        start_date = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2026, 12, 31, tzinfo=timezone.utc)

        events = fetch_events_for_agenda(123, start_date, end_date)

        print("\n✅ Filtre par période vérifié (timings conformes)")

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["uid"], 789)
        self.assertEqual(events[0]["parent_agenda_uid"], 123)

        # Vérifier que les paramètres de date sont correctement envoyés à l'API
        mock_get.assert_called_once()
        call_params = mock_get.call_args[1]["params"]
        self.assertEqual(call_params["timings[gte]"], "2026-01-01T00:00:00+00:00")
        self.assertEqual(call_params["timings[lte]"], "2026-12-31T00:00:00+00:00")

    def test_clean_and_save_events_processing(self):
        """
        Teste le processus de nettoyage : déduplication, filtrage des valeurs manquantes,
        et extraction du français.
        """
        # Événements fictifs bruts
        raw_events = [
            # Événement valide 1
            {
                "uid": 1001,
                "slug": "event-1",
                "title": {"fr": "Titre Français", "en": "English Title"},
                "description": {"fr": "Description Française"},
                "keywords": {"fr": ["mots", "clés"]},
                "location": {
                    "address": "1 Rue de Lyon",
                    "city": "Lyon",
                    "latitude": 45.76,
                    "longitude": 4.83,
                },
                "firstTiming": "2026-07-03",
                "lastTiming": "2026-07-04",
                "attendanceMode": 1,
            },
            # Événement 2 : Doublon sur l'UID 1001 (doit être filtré)
            {
                "uid": 1001,
                "title": {"fr": "Doublon Titre"},
                "location": {"city": "Lyon"},
            },
            # Événement 3 : Pas de location (doit être filtré)
            {
                "uid": 1002,
                "title": {"fr": "Événement sans adresse"},
                "location": None,
            },
            # Événement 4 : Pas de dictionnaire pour title/description (cas limites)
            {
                "uid": 1003,
                "title": "Titre simple string",
                "description": None,
                "location": {"city": "Grenoble"},
            },
        ]

        # Utiliser un fichier temporaire pour tester la sauvegarde
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_output_path = os.path.join(tmpdir, "test_clean.json")

            # Exécuter le nettoyage
            clean_and_save_events(raw_events, temp_output_path)

            print("\n✅ Nettoyage vérifié (champs présents, pas de doublons, pas de valeurs vides)")

            # Vérifier l'existence et lire le fichier de sortie
            self.assertTrue(os.path.exists(temp_output_path))
            with open(temp_output_path, "r", encoding="utf-8") as f:
                cleaned_data = json.load(f)

            # Vérifications sur le contenu nettoyé
            # On doit avoir conservé uniquement 2 événements (1001 et 1003)
            self.assertEqual(len(cleaned_data), 2)

            # Vérification de l'événement 1001 (champs en français extraits)
            ev1 = cleaned_data[0]
            self.assertEqual(ev1["uid"], 1001)
            self.assertEqual(ev1["title_fr"], "Titre Français")
            self.assertEqual(ev1["description_fr"], "Description Française")
            self.assertEqual(ev1["keywords_fr"], ["mots", "clés"])
            self.assertEqual(ev1["location"]["city"], "Lyon")

            # Vérification de l'événement 1003 (champs limites)
            ev3 = cleaned_data[1]
            self.assertEqual(ev3["uid"], 1003)
            self.assertEqual(ev3["title_fr"], "Titre simple string")
            self.assertEqual(ev3["description_fr"], "")
            self.assertEqual(ev3["keywords_fr"], [])
            self.assertEqual(ev3["location"]["city"], "Grenoble")


if __name__ == "__main__":
    unittest.main()
