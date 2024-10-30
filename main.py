import recherche_similarite
import gradio as gr

def repondre(question):
    embeddings_documents = recherche_similarite.chargement_embedding('document_embeddings.json')
    resultats = recherche_similarite.rechercher_documents_similaires(question, embeddings_documents)
    reponses = [f"ID: {doc_id}, Similarité: {score:.4f}, Texte: {texte}" for doc_id, texte, score in resultats]
    return "\n".join(reponses)

# Interface Gradio
interface = gr.Interface(
    fn=repondre,
    inputs="text",
    outputs="text",
    title="Recherche de Documents Similaires",
    description="Entrez votre question pour rechercher des documents similaires."
)

if __name__ == "__main__":
    interface.launch()