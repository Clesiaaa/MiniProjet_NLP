import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

def charger_doc(chemin:str)->list[str]:
    """charge un document .txt dans une liste"""
    with open(chemin, "r") as fichier:

        doc=[]

        for ligne in fichier:
            doc.append(ligne.strip())

    return doc

def encodage(documents:list[str], nom_du_fichier_de_sortie:str)->None:
    """encode et puis enregistre les embeddings dans un json"""

    embedding=model.encode(documents)

    taille=len(documents)
    i=0
    data=[]

    while i < taille:
        data.append({
        "id": i,
        "text": documents[i],
        "embedding": embedding[i].tolist()})
        i+=1
        
    with open(nom_du_fichier_de_sortie, "w") as fichier_sortie:
        json.dump(data, fichier_sortie, ensure_ascii=False, indent=4)

documents = charger_doc("data/donees.txt")
encodage(documents, 'json/document_embeddings.json')