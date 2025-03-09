import os
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import re

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


PROMPT_TEMPLATE = """<instructions>
You are an expert assistant. Your answer must rely exclusively on the information provided in the <context> section. Do not incorporate any external knowledge or details.
1. Use only the information provided in the <context> section. If the necessary details are not present, respond exactly: "I do not see that information in the context."
2. When referring to details in the context, indicate where the information is found (for example, "as noted in the context", page 26, etc) if it supports your answer.
3. For questions involving numerical values or any type of calculations, provide only the final answer without intermediate calculations. Ensure that all numbers are accompanied by clear and correct units (e.g., "million" or "billion") exactly as stated in the context, and double-check these details for accuracy.
4. Before answering, check the headings from **Table of contents** or **Table Data** for any additional unit specifications provided in brackets (for example, "(in millions)" or "(in billions)"). Use these units exactly as indicated. For instance, if the heading specifies that values are "(in millions)", your answer must reflect that unit.
5. Ensure that your final answer is factually correct and fully supported by the context. Do not include any extra commentary, page references, assumptions, inferences, or external data.
6. Be clear, concise, and accurate in your answer while maintaining a friendly, professional tone.
7. If the context provides ambiguous or conflicting information, explicitly state that the information is ambiguous.
8. Do not speculate or add any information that is not explicitly stated in the context. If a detail is missing, respond with: "I do not see that information in the context."
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


def merge_chunks(docs):
    """
      - Groups chunks by their page number and sorts pages in ascending order.
      - Within each page, orders chunks in their original order.
      - Attempts to remove overlapping duplicate sentences between consecutive chunks.
    """
    from collections import defaultdict
    page_map = defaultdict(list)

    # Group chunks by page number
    for d in docs:
        page = d.metadata.get("page", -1)
        page_map[page].append(d.page_content)

    merged_texts = []
    for page in sorted(page_map.keys()):
        chunks = page_map[page]
        deduped_chunks = []
        previous_chunk = ""
        for chunk in chunks:
            if previous_chunk:
                # Split previous chunk into sentences using a simple regex.
                prev_sentences = re.split(r'(?<=[.!?])\s+', previous_chunk.strip())
                if prev_sentences:
                    last_sentence = prev_sentences[-1]
                    # If the current chunk starts with the same sentence, remove the duplicate.
                    if chunk.startswith(last_sentence):
                        chunk = chunk[len(last_sentence):].strip()
            deduped_chunks.append(chunk)
            previous_chunk = chunk
        combined = "\n".join(deduped_chunks)
        merged_texts.append(f"(Page {page})\n{combined}")

    return "\n---\n".join(merged_texts)

def answer_query(user_message: str, history: list, collection_name: str):
    if not collection_name or not collection_name.strip():
        raise ValueError("Collection name must be a non-empty string")

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name.strip(),
        embeddings=embeddings,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
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

    prompt = PROMPT_TEMPLATE.format(
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