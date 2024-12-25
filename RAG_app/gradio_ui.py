import logging
import gradio as gr
from indexing import index_file_in_qdrant
from inference import answer_query
from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#show list of collection names from QDrant
def list_qdrant_collections(url="http://localhost:6333"):
    client = QdrantClient(url=url)
    collections = client.get_collections()
    collection_names = [c.name for c in collections.collections]
    logger.info(f"Retrieved collections: {collection_names}")
    return collection_names

def on_refresh_collections():
    return gr.update(choices=list_qdrant_collections())

def perform_index(pdf_file, coll_name):
    if not pdf_file:
        return "No PDF selected."
    if not coll_name or not coll_name.strip():
        return "Collection name must be provided."
    # Check if collection exists
    existing_collections = list_qdrant_collections()
    if coll_name in existing_collections:
        action = "adding to"
        logger.info(f"Collection '{coll_name}' already exists. Adding documents to it.")
    else:
        action = "creating and indexing into"
        logger.info(f"Creating and indexing into new collection '{coll_name}'.")

    # Index the PDF
    msg = index_file_in_qdrant(pdf_file.name, coll_name)
    return f"Successfully {action} collection '{coll_name}' with '{pdf_file.name}'."

#inference part to get query and retrieve answer
def chat_fn(user_message, history, selected_coll):
    try:
        if not selected_coll:
            return "Please select a collection."
        return answer_query(user_message, history, selected_coll)
    except Exception as e:
        logger.exception("Chat error:")
        return f"Error: {e}"

def build_app():
    with gr.Blocks(title="RAG Demo") as demo:
        gr.Markdown("# RAG application\nSelect or create a collection(index a PDF in left column), then ask questions in RAG Chatbot.")

        with gr.Row():
            # QDrant Interface
            with gr.Column():
                gr.Markdown("### List of QDrant collection")

                init_collections = list_qdrant_collections()

                #selecting collections
                coll_dropdown = gr.Dropdown(
                    choices=init_collections,
                    label="Select Qdrant Collection",
                    value=init_collections[0] if init_collections else None,
                    interactive=True
                )
                refresh_btn = gr.Button("Refresh Collections")   #button to refresh collections list

                refresh_btn.click(
                    fn=on_refresh_collections,
                    outputs=coll_dropdown
                )
            #index part
            with gr.Column():
                gr.Markdown("### Index a new PDF")
                pdf_uploader = gr.File(file_types=[".pdf", ".docx"], label="PDF File")
                new_collection = gr.Textbox(label="Create new or change existing collection")
                index_btn = gr.Button("Index PDF")
                index_status = gr.Markdown()

                # Indexing action
                index_btn.click(
                    fn=perform_index,
                    inputs=[pdf_uploader, new_collection],
                    outputs=index_status
                )
        with gr.Row():
            with gr.Column(scale=12):
                # gr.Markdown("### RAG Chatbot")

                # Chat interface
                def chat_wrapper(msg, hist):
                    return chat_fn(msg, hist, coll_dropdown.value)

                example_questions = [
                    "What position in the company does Jeffrey P. Bezos hold and since when?",
                    "When was the company founded?"
                ]
                chatbot = gr.ChatInterface(
                    fn=chat_wrapper,
                    type="messages",
                    examples=example_questions,
                    title="RAG Chatbot",
                    description="Choose your QDrant collection or create a new one on the right."
                )
    return demo

if __name__ == "__main__":
    app = build_app()
    app.launch()