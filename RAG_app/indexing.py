import os
import logging
import pdfplumber
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_documents_from_pdf(pdf_path):
    def extract_tables_as_text(pdf_page):
        tables = pdf_page.extract_tables()
        table_texts = []
        for table in tables:
            row_strings = []
            for row in table:
                cleaned = [cell if cell is not None else "" for cell in row]
                row_strings.append(", ".join(cleaned))
            table_texts.append("\n".join(row_strings))
        return "\n\n".join(table_texts)

    documents = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            table_text = extract_tables_as_text(page)
            if table_text.strip():
                text += "\n\n[Table Data]\n" + table_text

            if text.strip():
                documents.append(Document(page_content=text))
    return documents

#create qdrant coolection if it not exist
def create_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int = 384, distance: str = "Cosine"):
    try:
        client.get_collection(collection_name)
        logger.info(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        logger.info(f"Creating collection '{collection_name}'...")
        # Define the distance metric using the Distance enum
        distance_metric = Distance.COSINE if distance.lower() == "cosine" else Distance.EUCLID
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance_metric
            )
        )
        logger.info(f"Collection '{collection_name}' created successfully.")

def index_file_in_qdrant(pdf_path: str, collection_name: str, qdrant_url="http://localhost:6333") -> str:
    if not pdf_path or not os.path.exists(pdf_path):
        raise ValueError(f"PDF file does not exist: {pdf_path}")
    if not collection_name or not collection_name.strip():
        raise ValueError("Collection name must be specified.")

    logger.info(f"Indexing PDF '{pdf_path}' into collection '{collection_name}'")

    docs = load_documents_from_pdf(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    logger.info(f"Split into {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    client = QdrantClient(url=qdrant_url)

    create_qdrant_collection(client, collection_name, vector_size=384, distance="Cosine")

    qdrant_store = Qdrant(
        client=client,
        collection_name=collection_name.strip(),
        embeddings=embeddings,
    )
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    # metadatas = [{}] * len(chunks)

    qdrant_store.add_texts(texts=texts, metadatas=metadatas)

    msg = f"Successfully indexed '{os.path.basename(pdf_path)}' into collection '{collection_name}'!"
    logger.info(msg)
    return msg      #string