import gradio as gr
from recherche_similarite import chargement_embedding, rechercher_documents_similaires

embeddings_documents = chargement_embedding("json/document_embeddings.json")

def chat(question:str)->str:
    """Apporte la réponse par rapport à la question posé"""
    resultats = rechercher_documents_similaires(question, embeddings_documents)
    
    meilleur_resultat = resultats[0]
    reponse_text = meilleur_resultat[1]
    score = meilleur_resultat[2]
    
    return f"Réponse : {reponse_text}\nScore : {score:.4f}"

#source: https://www.youtube.com/watch?v=H7JDoS4vLMU, https://www.youtube.com/watch?v=eE7CamOE-PA&t=395s, chatgpt
interface = gr.Interface(
    fn=chat,
    inputs="text",
    outputs="text",
    title="MiniNLP",
    description="Posez une question et recevez la réponse la plus pertinente avec un score de similarité",
)

if __name__ == "__main__":
    interface.launch()
