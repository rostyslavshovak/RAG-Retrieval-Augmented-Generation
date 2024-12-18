import os
import argparse
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from RAG_QDrant.rag_script.utils import load_documents_from_pdf

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

def main():
    #Load OpenAI API key from .env
    load_dotenv()
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found. Set it in the environment or .env file.")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run Naive RAG Pipeline on a PDF file.")
    parser.add_argument("--pdf_path", required=True, help="Path to the PDF file.")
    parser.add_argument("--query", required=True, help="Question to query based on the file.")
    args = parser.parse_args()

    pdf_path = args.pdf_path        # arguments
    query = args.query

    #load documents
    documents = load_documents_from_pdf(args.pdf_path)

    #split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)
    print(f"Split documents into {len(docs)} chunks.")

    #Embeddings
    embedding_model_name = "sentence-transformers/multi-qa-MiniLM-L6-dot-v1"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    print(f"Loaded HuggingFace Embeddings model: {embeddings_model_name}")

    # If Qdrant is running locally via Docker
    # docker run -p 6333:6333 qdrant/qdrant
    qdrant_client = QdrantClient(url="http://localhost:6333")
    collection_name = "rag"

    # Create Qdrant vector store
    vectorstore = Qdrant.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        url="http://localhost:6333",
        force_recreate=True,            #recreate the vector store if dimension is diff
        prefer_grpc=False,              # Protocol Buffers (protobuf) for data serialization, which is compact and faster compared to JSON
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5, "score_threshold": 0.5})

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=500,
        openai_api_key=OPENAI_API_KEY
    )

    prompt_template = """You are a helpful assistant. Using the provided context, write a complete, well-structured, and detailed answer to the question. Incorporate any relevant information from the context into your answer. Be direct, accurate, and friendly in tone. Provide dates, positions, and context as necessary.
    If you cannot find the answer in the context, say you don't know.
    
    Context:
    {context}

    Question: {question}

    Please provide a detailed and well-structured answer:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    #Retrieve documents and show retrieved chunks
    retrieved_docs = retriever.invoke(query)
    retrieved_chunks = [doc.page_content for doc in retrieved_docs]

    response = qa_chain.invoke({"query": query})
    answer = response["result"]

    print("\nQuestion:", query)
    print("\nAnswer:", answer)
    print("\n\nRetrieved Chunks:")
    for i, chunk in enumerate(retrieved_chunks, start=1):
        print(f"Chunk {i}:\n{chunk}\n{'-' * 40}")

if __name__ == "__main__":
    main()