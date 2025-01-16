import os
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient

from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv('MODEL_NAME')
TEMPERATURE = os.getenv('TEMPERATURE')
MAX_TOKENS = os.getenv('MAX_TOKENS')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
QDRANT_PORT = os.getenv('PORT')
QDRANT_HOST = os.getenv('HOST')

def merge_chunks(docs):
    from collections import defaultdict
    page_map = defaultdict(list)
    for d in docs:
        page = d.metadata.get("page", -1)
        page_map[page].append(d.page_content)

    merged_texts = []
    for page, texts in page_map.items():
        combined = "\n".join(texts)
        merged_texts.append(f"(Page {page})\n{combined}")

    return "\n---\n".join(merged_texts)

def answer_query(user_message: str, history: list, collection_name: str):
    if not collection_name or not collection_name.strip():
        return "No Qdrant collection selected.", ""

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )

    qdrant_url = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
    client = QdrantClient(url=qdrant_url)

    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name.strip(),
        embeddings=embeddings,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    raw_docs = retriever.get_relevant_documents(user_message)

    merged_content = merge_chunks(raw_docs)

    conversation_so_far = []
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            conversation_so_far.append(f"User: {content}")
        elif role == "assistant":
            conversation_so_far.append(f"Assistant: {content}")

    chat_history_text = "\n".join(conversation_so_far)

    prompt_template = """<instructions>
    You are a knowledgeable assistant with access to user-provided context.
    Use only the retrieved context below to answer the user’s question or provide guidance.
    Cite relevant sections from context. Do not add info that is not in the context.
    If you do not see relevant details in the context, say "I you do not see that information in the context".
    </instructions>
    
    <context>
    {context}
    </context>
    
    <conversation_history>
    {chat_history}
    </conversation_history>
    
    <question>
    {question}
    </question>
    
    <answer>
    """
    prompt = prompt_template.format(
        context=merged_content,
        chat_history=chat_history_text,
        question=user_message
    )

    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        temperature=float(TEMPERATURE),
        max_tokens=int(MAX_TOKENS),
        openai_api_key=OPENAI_API_KEY
    )
    response = llm.predict(prompt)
    final_answer = response.strip() + "\n</answer>"

    return final_answer, merged_content