import os
import sys
import json
import time
from datetime import datetime, timedelta, timezone
import requests
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Récupérer la clé API d'OpenAgenda
API_KEY = os.getenv("OPENAGENDA_API_KEY")
if not API_KEY:
    print("Erreur : La variable d'environnement OPENAGENDA_API_KEY n'est pas définie.")
    sys.exit(1)

# Configuration de base pour l'API
BASE_URL = "https://api.openagenda.com/v2"
HEADERS = {
    "key": API_KEY,
    "Content-Type": "application/json"
}

def fetch_all_agendas(search_query):
    """
    Récupère tous les agendas contenant la chaîne de recherche spécifiée.
    Gère la pagination par curseur ('after') pour récupérer l'intégralité des résultats.
    """
    print(f"Recherche des agendas contenant : '{search_query}'...")
    agendas = []
    after = None
    page = 1

    while True:
        params = {
            "search": search_query,
            "size": 100  # Taille maximale autorisée par page pour optimiser le nombre de requêtes
        }
        if after:
            params["after[]"] = after

        try:
            response = requests.get(f"{BASE_URL}/agendas", params=params, headers=HEADERS)
            if response.status_code != 200:
                print(f"Erreur lors de la récupération des agendas (page {page}) : Code {response.status_code}")
                print(response.text)
                break
            
            data = response.json()
            page_agendas = data.get("agendas", [])
            agendas.extend(page_agendas)
            
            print(f"Page {page} : {len(page_agendas)} agendas récupérés (Total : {len(agendas)})")
            
            # Récupérer le curseur pour la page suivante
            after = data.get("after")
            if not after or not page_agendas:
                # Si plus aucun curseur 'after' ou page vide, on a fini
                break
                
            page += 1
            time.sleep(0.1) # Légère pause pour respecter les limites de l'API
            
        except Exception as e:
            print(f"Exception lors de la récupération des agendas : {e}")
            break

    return agendas

def fetch_events_for_agenda(agenda_uid, start_date, end_date):
    """
    Récupère tous les événements d'un agenda spécifique sur une plage de dates donnée.
    Gère la pagination par curseur pour cet agenda.
    """
    events = []
    after = None
    
    while True:
        params = {
            "timings[gte]": start_date.isoformat(),
            "timings[lte]": end_date.isoformat(),
            "size": 100  # Taille maximale autorisée par page
        }
        if after:
            params["after[]"] = after

        try:
            url = f"{BASE_URL}/agendas/{agenda_uid}/events"
            response = requests.get(url, params=params, headers=HEADERS)
            if response.status_code != 200:
                # Si l'agenda n'est pas accessible ou une autre erreur survient, on l'affiche et passe
                print(f"  [Agenda {agenda_uid}] Erreur {response.status_code} lors de la récupération des événements.")
                break
            
            data = response.json()
            page_events = data.get("events", [])
            for event in page_events:
                # Ajouter l'UID de l'agenda parent dans chaque événement pour traçabilité
                event["parent_agenda_uid"] = agenda_uid
                events.append(event)
            
            after = data.get("after")
            if not after or not page_events:
                break
                
            time.sleep(0.1)
            
        except Exception as e:
            print(f"  [Agenda {agenda_uid}] Exception lors de la récupération des événements : {e}")
            break
            
    return events

def main():
    # Définir l'intervalle de temps (12 prochains mois)
    now = datetime.now(timezone.utc)
    twelve_months_later = now + timedelta(days=365)
    one_year_ago = now - timedelta(days=365)

    
    print(f"Période de recherche d'événements :")
    print(f"- Début : {one_year_ago.isoformat()}")
    print(f"- Fin   : {twelve_months_later.isoformat()}")
    print("-" * 50)
    
    # 1. Rechercher tous les agendas d'Auvergne-Rhône-Alpes
    agendas = fetch_all_agendas("auvergne-rhone-alpes")
    if not agendas:
        print("Aucun agenda trouvé.")
        return
        
    uids = [agenda["uid"] for agenda in agendas]
    print(f"Nombre total d'agendas récupérés : {len(uids)}")
    print("-" * 50)
    
    # 2 & 3. Récupérer les événements pour chaque agenda
    all_events = []
    for idx, uid in enumerate(uids, start=1):
        print(f"[{idx}/{len(uids)}] Récupération des événements pour l'agenda UID {uid}...")
        agenda_events = fetch_events_for_agenda(uid, one_year_ago, twelve_months_later)
        if agenda_events:
            all_events.extend(agenda_events)
            print(f"  -> {len(agenda_events)} événements trouvés. (Total cumulé : {len(all_events)})")
        else:
            print("  -> Aucun événement à venir trouvé.")
            
        # Pause légère pour ne pas surcharger l'API d'OpenAgenda
        time.sleep(0.1)
        
    print("-" * 50)
    print(f"Récupération terminée. Nombre total d'événements : {len(all_events)}")
    
    # 4. Sauvegarder le résultat dans data/events_raw.json
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "events_raw.json")
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_events, f, ensure_ascii=False, indent=2)
        print(f"Succès : Les événements ont été sauvegardés dans '{output_path}'.")
    except Exception as e:
        print(f"Erreur lors de l'écriture du fichier de sortie : {e}")

if __name__ == "__main__":
    main()
