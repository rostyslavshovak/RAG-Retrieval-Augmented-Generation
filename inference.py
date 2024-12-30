import os
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

MODEL_NAME = os.getenv('MODEL_NAME')
TEMPERATURE = os.getenv('TEMPERATURE')
MAX_TOKENS = os.getenv('MAX_TOKENS')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
QDRANT_PORT = os.getenv('PORT')
QDRANT_HOST = os.getenv('HOST')

def answer_query(user_message: str, history: list, collection_name: str) -> str:
    if not collection_name or not collection_name.strip():
        return "No Qdrant collection selected."

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    qdrant_url = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
    client = QdrantClient(url=qdrant_url)

    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name.strip(),
        embeddings=embeddings,
        # prefer_grpc=False
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6, "score_threshold": 0.5})

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set in environment or .env")

    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        openai_api_key=OPENAI_API_KEY
    )
    prompt_template = """You are a helpful assistant. Using the provided context, write a complete, well-structured, and detailed answer to the question. Incorporate any relevant information from the context into your answer. Be direct, accurate, and friendly in tone. Provide dates, positions, and context as necessary.
            If you do not see relevant information in the context, say you don't have enough information.

            Context:
            {context}

            Question: {question}

            Answer with precision:
            """
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        }
    )
    result = chain({"query": user_message})
    return result["result"]