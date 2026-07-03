import json
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS

def build_index():
    # 1. Charger les variables d'environnement (MISTRAL_API_KEY)
    load_dotenv()
    
    if not os.getenv("MISTRAL_API_KEY"):
        print("Erreur: MISTRAL_API_KEY n'est pas définie dans le fichier .env")
        return
        
    # 2. Charger les événements depuis events_clean.json
    print("Chargement des événements depuis data/events_clean.json...")
    input_path = os.path.join("data", "events_clean.json")
    if not os.path.exists(input_path):
        print(f"Erreur: Le fichier {input_path} est introuvable.")
        return
        
    with open(input_path, "r", encoding="utf-8") as f:
        events = json.load(f)
        
    print(f"{len(events)} événements chargés. Création des chunks...")
    
    # 3. Transformer chaque événement en un chunk texte
    documents = []
    for event in events:
        # Extraire la ville
        city = ""
        loc = event.get("location")
        if isinstance(loc, dict):
            city = loc.get("city", "")
        elif isinstance(loc, str):
            city = loc
            
        # Gérer les mots-clés
        keywords = event.get("keywords_fr", [])
        keywords_str = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)
        
        # Concaténer selon les instructions : title, description, keywords, ville, firstTiming
        text_chunk = (
            f"Titre : {event.get('title_fr', '')}\n"
            f"Description : {event.get('description_fr', '')}\n"
            f"Mots-clés : {keywords_str}\n"
            f"Ville : {city}\n"
            f"Date de début : {event.get('firstTiming', '')}"
        )
        
        # On attache l'UID en métadonnée pour pouvoir faire le lien avec la base de données json plus tard
        metadata = {
            "uid": event.get("uid"),
        }
        
        doc = Document(page_content=text_chunk, metadata=metadata)
        documents.append(doc)
        
    # 4. Utiliser Mistral pour générer les embeddings
    print("Initialisation du modèle d'embedding Mistral...")
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    
    # 5. Stocker tous les vecteurs dans FAISS
    print("Génération des vecteurs et construction de l'index FAISS (peut prendre quelques secondes/minutes)...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # 6. Sauvegarder l'index FAISS sur le disque
    output_dir = os.path.join("data", "faiss_index")
    print(f"Sauvegarde de l'index FAISS dans {output_dir}...")
    vectorstore.save_local(output_dir)
    
    print("Succès ! L'index FAISS a été construit et sauvegardé.")

if __name__ == "__main__":
    build_index()
