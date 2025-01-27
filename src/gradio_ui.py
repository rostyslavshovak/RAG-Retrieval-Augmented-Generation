import os
from dotenv import load_dotenv
import logging
import warnings
import gradio as gr
from qdrant_client import QdrantClient

from src.indexing import index_file_in_qdrant
from src.inference import answer_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=DeprecationWarning)

load_dotenv()
QDRANT_HOST = os.getenv('HOST')
QDRANT_PORT = os.getenv('PORT')

def list_qdrant_collections():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return [c.name for c in client.get_collections().collections]

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

    existing_collections = list_qdrant_collections()
    if coll_name in existing_collections:
        action = "adding to"
        logger.info(f"Collection '{coll_name}' already exists. Adding documents to it.")
    else:
        action = "creating and indexing into"
        logger.info(f"Creating and indexing into new collection '{coll_name}'.")

    msg = index_file_in_qdrant(pdf_file.name, coll_name)
    new_list = list_qdrant_collections()
    logger.info(f"Updated collections after indexing: {new_list}")

    return f"Successfully {action} collection '{coll_name}'.", gr.update(choices=new_list, value=coll_name)

def chat_fn(user_message, history, selected_coll):
    """
    Called on each new user message.
    'history' is a list of dicts ( [ {"role":"user","content":...}, {"role":"assistant","content":...} ] ).
    Returns the LLM answer + merged chunks.
    """
    try:
        if not selected_coll:
            return "Please select a collection.", ""
        ans, retrieved_text = answer_query(user_message, history, selected_coll)
        return ans, retrieved_text
    except Exception as e:
        logger.exception("Chat error:")
        return f"Error: {e}", ""

def respond(message, history, selected_coll):
    logger.info(f"Received message: {message}")
    logger.info(f"Current history: {history}")
    logger.info(f"Selected collection: {selected_coll}")

    history = history + [{"role": "user", "content": message}]

    llm_answer, retrieved_text = chat_fn(message, history, selected_coll)

    updated_history = history + [{"role": "assistant", "content": llm_answer}]

    return updated_history, updated_history, retrieved_text

def build_app():
    retrieved_chunks_box = gr.Textbox(
        label="Retrieved & Merged Context",
        lines=20,
        interactive=False,
        placeholder="Retrieved context will appear here after each question."
    )

    with gr.Blocks(title="RAG Chatbot") as demo:
        gr.Markdown("# RAG Application\n")
        gr.Markdown("Select or create a collection in the left column, then ask questions in the Chatbot tab.\n")

        with gr.Row():
            with gr.Column():
                gr.Markdown("## QDrant Collections")
                init_collections = list_qdrant_collections()
                coll_dropdown = gr.Dropdown(
                    choices=init_collections,
                    label="Select Qdrant Collection",
                    value=init_collections[0] if init_collections else None,
                    interactive=True
                )
                refresh_btn = gr.Button("Refresh Collections")
                refresh_btn.click(
                    fn=on_refresh_collections,
                    outputs=coll_dropdown
                )

            with gr.Column():
                gr.Markdown("## Index a new PDF")
                pdf_uploader = gr.File(file_types=[".pdf", ".docx"], label="PDF File")
                new_collection = gr.Textbox(label="Create or enter an existing collection")
                index_btn = gr.Button("Index PDF")
                index_status = gr.Markdown()

                index_btn.click(
                    fn=perform_index,
                    inputs=[pdf_uploader, new_collection],
                    outputs=[index_status, coll_dropdown],
                    queue=False
                )

        with gr.Tab("Chatbot"):
            chatbot = gr.Chatbot(label="RAG Chatbot", type="messages")
            message_input = gr.Textbox(
                placeholder="Type your message here...",
                label="Your Message",
                lines=1
            )
            submit_btn = gr.Button("Send")

            # Keep conversation as a list of dicts: e.g. [{"role":"user","content":"..."}, ...]
            state = gr.State([])

            submit_btn.click(
                fn=respond,
                inputs=[message_input, state, coll_dropdown],
                outputs=[chatbot, state, retrieved_chunks_box],
            )
            message_input.submit(
                fn=respond,
                inputs=[message_input, state, coll_dropdown],
                outputs=[chatbot, state, retrieved_chunks_box],
            )

            gr.Examples(
                examples=[
                    "What was Amazon’s net cash provided by operating activities in 2019?",
                    "When was Amazon founded?"
                ],
                inputs=[message_input],
                label="Example Questions"
            )

        with gr.Tab("Retrieved Chunks"):
            gr.Markdown("### Merged Chunks for the Latest Question")
            retrieved_chunks_box.render()

    return demo

if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7861)