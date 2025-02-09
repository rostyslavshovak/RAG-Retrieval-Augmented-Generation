import os
from dotenv import load_dotenv
import logging
import pdfplumber
from langchain_community.docstore.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

import nltk
from typing import List

#Download the punkt tokenizer (if not already downloaded)
nltk.download('punkt', quiet=True)


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_PORT = os.getenv('PORT')
QDRANT_HOST = os.getenv('HOST')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# CHUNK_SPLITTER_CONFIG = {
#     "chunk_size": 1200,
#     "chunk_overlap": 240,
#     "separators": ["\n\n## ", "\n\n• ", "\n\n", "\n", ". "]
# }

TEXT_SPLITTER_CONFIG = {
        "chunk_size": 10,
        "chunk_overlap": 3
}

class SentenceWindowTextSplitter:
    def __init__(self, chunk_size: int = 10, chunk_overlap: int = 3):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        sentences = nltk.sent_tokenize(text)
        chunks = []
        index = 0
        while index < len(sentences):
            #Create a window of sentences
            window = sentences[index: index + self.chunk_size]
            #Join the sentences to form a text chunk
            chunk = " ".join(window)
            chunks.append(chunk)
            index += (self.chunk_size - self.chunk_overlap)
        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        new_documents = []
        for doc in documents:
            splits = self.split_text(doc.page_content)
            for split in splits:
                new_doc = Document(page_content=split, metadata=doc.metadata)
                new_documents.append(new_doc)
        return new_documents

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
        for page_idx, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            table_text = extract_tables_as_text(page)

            if table_text.strip():
                text += "\n\n[Table Data]\n" + table_text

            if text.strip():
                metadata = {"page": page_idx + 1}
                documents.append(Document(page_content=text, metadata=metadata))
    return documents

def create_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int = 1536, distance: str = "Cosine"):
    try:
        client.get_collection(collection_name)
        logger.info(f"Collection '{collection_name}' already exists.")
    except Exception:
        logger.info(f"Creating collection '{collection_name}'...")
        distance_metric = Distance.COSINE if distance.lower() == "cosine" else Distance.EUCLID
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance_metric
            )
        )
        logger.info(f"Collection '{collection_name}' created successfully.")

def index_file_in_qdrant(pdf_path: str, collection_name: str) -> str:
    if not pdf_path or not os.path.exists(pdf_path):
        raise ValueError(f"PDF file does not exist: {pdf_path}")
    if not collection_name or not collection_name.strip():
        raise ValueError("Collection name must be specified.")

    logger.info(f"Indexing PDF '{pdf_path}' into collection '{collection_name}'...")

    docs = load_documents_from_pdf(pdf_path)

    #char-based approach:
    # splitter = RecursiveCharacterTextSplitter(**CHUNK_SPLITTER_CONFIG)
    # chunks = splitter.split_documents(docs)
    # logger.info(f"Split into {len(chunks)} chunks.")

    splitter = SentenceWindowTextSplitter(**TEXT_SPLITTER_CONFIG)
    chunks = splitter.split_documents(docs)
    logger.info(f"Split into {len(chunks)} chunks using sentence-window retrieval.")


    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    create_qdrant_collection(client, collection_name, vector_size=1536, distance="Cosine")

    qdrant_store = Qdrant(
        client=client,
        collection_name=collection_name.strip(),
        embeddings=embeddings,
    )
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    qdrant_store.add_texts(texts=texts, metadatas=metadatas)

    msg = f"Successfully indexed '{os.path.basename(pdf_path)}' into collection '{collection_name}'!"
    logger.info(msg)
    return msg