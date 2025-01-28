import os
from dotenv import load_dotenv
import logging
import warnings
import gradio as gr
from indexing import index_file_in_qdrant
from inference import answer_query
from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=DeprecationWarning)

load_dotenv()
QDRANT_HOST = os.getenv('HOST')
QDRANT_PORT = os.getenv('PORT')

#show list of collection names from QDrant
def list_qdrant_collections():
    qdrant_url = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
    client = QdrantClient(url=qdrant_url)

    collections = client.get_collections()
    collection_names = [c.name for c in collections.collections]
    logger.info(f"Retrieved collections: {collection_names}")
    return collection_names

def on_refresh_collections():
    updated_collections = list_qdrant_collections()
    logger.info(f"Refreshing collections: {updated_collections}")
    if updated_collections:
        return gr.update(choices=updated_collections, value=updated_collections[0])
    else:
        return gr.update(choices=[], value=None)

def perform_index(pdf_file, coll_name):
    if not pdf_file:
        return ("No PDF selected.", gr.update())

    if not coll_name or not coll_name.strip():
        return ("Collection name must be provided.", gr.update())

    # Check if collection exists
    existing_collections = list_qdrant_collections()
    if coll_name in existing_collections:
        action = "adding to"
        logger.info(f"Collection '{coll_name}' already exists. Adding documents to it.")
    else:
        action = "creating and indexing into"
        logger.info(f"Creating and indexing into new collection '{coll_name}'.")

    msg = index_file_in_qdrant(pdf_file.name, coll_name)        #index PDF file
    new_list = list_qdrant_collections()

    #update list of collections and in return choose the created collection
    logger.info(f"Updated collections after indexing: {new_list}")
    return f"Successfully {action} collection '{coll_name}'.", gr.update(choices=new_list, value=coll_name)

#inference part to get query and retrieve answer
def chat_fn(user_message, history, selected_coll):
    try:
        if not selected_coll:
            return "Please select a collection."
        return answer_query(user_message, history, selected_coll)
    except Exception as e:
        logger.exception("Chat error:")
        return f"Error: {e}"

def respond(message, history, selected_coll):
    logger.info(f"Received message: {message}")
    logger.info(f"Current history: {history}")
    logger.info(f"Selected collection: {selected_coll}")

    response = chat_fn(message, history, selected_coll)

    updated_history = history + [[message, response]]
    return updated_history, updated_history

def build_app():
    with gr.Blocks(title="RAG Demo") as demo:
        gr.Markdown("# RAG application\nSelect or create a collection (index a PDF in left column), then ask questions in RAG Chatbot.")

        with gr.Row():
            #QDrant Interface
            with gr.Column():
                gr.Markdown("## List of QDrant collection")

                init_collections = list_qdrant_collections()

                #selecting collections
                coll_dropdown = gr.Dropdown(
                    choices=init_collections,
                    label="Select Qdrant Collection",
                    value=init_collections[0] if init_collections else None,
                    interactive=True
                )
                refresh_btn = gr.Button("Refresh Collections")  #button to refresh collections list

                refresh_btn.click(
                    fn=on_refresh_collections,
                    outputs=coll_dropdown
                )
            #index part
            with gr.Column():
                gr.Markdown("## Index a new PDF")
                pdf_uploader = gr.File(file_types=[".pdf", ".docx"], label="PDF File")
                new_collection = gr.Textbox(label="Create new or change existing collection")
                index_btn = gr.Button("Index PDF")
                index_status = gr.Markdown()

                #Indexing action and update dropdown
                index_btn.click(
                    fn=perform_index,
                    inputs=[pdf_uploader, new_collection],
                    outputs=[index_status, coll_dropdown],
                    queue=False
                )
        with gr.Row():
            with gr.Column(scale=12):
                gr.Markdown("## RAG Chatbot")

                chatbot = gr.Chatbot(label="RAG Chatbot")
                message_input = gr.Textbox(
                    placeholder="Type your message here...",
                    label="Your Message",
                    lines=1
                )
                submit_btn = gr.Button("Send")

                state = gr.State([])    #keep history of conversation

                submit_btn.click(
                    fn=respond,
                    inputs=[message_input, state, coll_dropdown],
                    outputs=[chatbot, state],
                )

                message_input.submit(           #allow to send message by pressing Enter
                    fn=respond,
                    inputs=[message_input, state, coll_dropdown],
                    outputs=[chatbot, state],
                )
                gr.Examples(
                    examples=[
                        "What was Amazon’s net cash provided by operating activities in 2019?",
                        "When was Amazon founded?"
                    ],
                    inputs=[message_input],
                    label="Example Questions"
                )
    return demo

if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)