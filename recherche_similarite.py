from sentence_transformers import SentenceTransformer
import json
import math

modele = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

def triage(liste:list)->list:
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

def chargement_embedding(chemin_fichier:str)->list:
    """Charge les embeddings"""
    with open(chemin_fichier, 'r') as fichier:
        donnees = json.load(fichier)
    return donnees

def calculer_similarite_cosinus(vecteur1: list, vecteur2: list) -> float:
    """Calcule la similarité cosinus entre deux vecteurs
    source : 
    https://www.youtube.com/watch?v=e9U0QAFbfLI
    https://www.youtube.com/watch?v=y-EjAuWdZdI
    https://www.youtube.com/watch?v=m_CooIRM3UI
    https://medium.com/@santannalouis208/la-similarité-cosinus-en-ia-nlp-d554d3b14efa
    """

    produit_scalaire = 0
    
    for x, y in zip(vecteur1, vecteur2):
        produit_scalaire += x*y

    somme_carres_vecteur1 = 0
    
    for x in vecteur1:
        
        somme_carres_vecteur1 += x ** 2
    norme_vecteur1 = math.sqrt(somme_carres_vecteur1)

    somme_carres_vecteur2 = 0
    
    for y in vecteur2:
        somme_carres_vecteur2 += y ** 2
        
    norme_vecteur2 = math.sqrt(somme_carres_vecteur2)

    if norme_vecteur1 == 0 or norme_vecteur2 == 0:
        return 0.0

    return produit_scalaire / (norme_vecteur1 * norme_vecteur2)


def rechercher_documents_similaires(question:str, embeddings_documents:list)->list:
    """Recherche la similarité la plus pertinante"""
    embedding_requete = modele.encode([question])[0]
    similarites = []
    
    for doc in embeddings_documents:
        score_similarite = calculer_similarite_cosinus(embedding_requete, doc["embedding"])
        similarites.append((doc["id"], doc["text"], score_similarite))
        
    resultats_tries = triage(similarites)
    
    return resultats_tries
