from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import json

modele = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

def triage(liste: list) -> list:
    """Le tri par insertion"""
    N = len(liste)
    for TC in range(1, N):
        temp = liste[TC]
        pos = TC
        while pos > 0 and liste[pos - 1][2] < temp[2]:
            liste[pos] = liste[pos - 1]
            pos -= 1
        liste[pos] = temp
    return liste

def chargement_embedding(chemin_fichier: str) -> list:
    """Charge les embeddings"""
    with open(chemin_fichier, 'r') as fichier:
        donnees = json.load(fichier)
    return donnees

def rechercher_documents_similaires(question, embeddings_documents):
    embedding_requete = modele.encode([question])[0]  # Encode the question

    similarites = []
    for doc in embeddings_documents:
        # Calculate the similarity score between the request embedding and the document's embedding
        score_similarite = 1 - cosine(embedding_requete, doc["embedding"])
        similarites.append((doc["id"], doc["text"], score_similarite))  # Collecting results

    resultats_tries = triage(similarites)  # Sort the results by similarity score
    return resultats_tries

# Exemple d'utilisation
embeddings_documents = chargement_embedding('json/document_embeddings.json')
question = "Comment est le nouveau téléphone ?"
tous_les_documents = rechercher_documents_similaires(question, embeddings_documents)

# Afficher les résultats
print("Documents les plus pertinents :")
res={}
for doc_id, texte, score in tous_les_documents:
    #print(f"ID: {doc_id}, Similarité: {score:.4f}, Texte: {texte}")
    res[score]=texte
meilleur_score = sorted(res.keys(), reverse=True)[0]
print(res[meilleur_score])