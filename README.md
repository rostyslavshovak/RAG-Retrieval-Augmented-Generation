# RAG Chatbot for Amazon 10-K Financial Analysis

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot designed for analyzing **Amazon's 10-K** financial reports. The chatbot helps users efficiently obtain accurate, detailed insights from these documents.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Enjoy the Chatbot!](#enjoy-the-chatbot)

---

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is an approach that combines information retrieval with the generative capabilities of Large Language Models. It works in three simple steps:

1. **Retrieve**: Extract documents from a knowledge base.
2. **Augment**: Add relevant context to the user prompt.
3. **Generate**: Generate an answer using the language model.

This method is particularly effective for domain-specific queries, ensuring responses are grounded in up-to-date and relevant information.

---

## Project Overview

The chatbot integrates the following key components to ensure reliable performance:

- **QDrant**: Serves as the vector database for document indexing, retrieval, and semantic search.
- **LangChain-Community**: Handles chunk splitting, embeddings, and chat model integration.
- **OpenAI Embeddings and Chat model**: Powers answer generation.
- **Gradio**: Provides an easy-to-use interface for document uploads and question-answering sessions.
- **RAGAS (Retrieval-Augmented Generation Assessment Score)**: Evaluates chatbot responses using detailed performance metrics.

The application is containerized with Docker, ensuring consistent deployment setup.

---

## Key Features

1. **Document Upload and Indexing**
   - Upload PDF files, which are then split and embedded using `OpenAIEmbeddings`.

2. **Vector Store with QDrant**
   - Indexed documents are stored in QDrant collections, enabling fast semantic search.

3. **Gradio UI**
   - User-friendly chat interface that allows you to:
     - Select or create QDrant collections.
     - Ask questions with real-time references to the source text.
     - View retrieved text chunks, ensuring transparency on the source of each answer.

4. **RAGAS Evaluation**
   - For details on the RAGAS evaluation methodology and results, see the [README](src/README.md).

5. **Sentence-Window Retrieval**
   - **Functionality**: Splits documents into overlapping windows based on sentences rather than arbitrary chunks.
   - **Benefits**: Enhances retrieval accuracy by preserving document context at the sentence level.

6. **Auto-merging Retrieval**
   - **Benefits**: Dynamically merges relevant retrieved information, delivering coherent and context-rich answers.

7. **Conversational Memory**
   - **Benefits**: Retains the context from previous interactions to support natural and continuous conversational flows.

8. **Unit Tests**
   - Ensures core functionalities (document indexing, retrieval, and answer generation) work as expected.

---

## Installation

Follow these steps to set up the project:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/rostyslavshovak/RAG-Retrieval-Augmented-Generation.git
   cd RAG-Retrieval-Augmented-Generation


2. **Install required dependencies**:
   ```bash
   pip install --no-cache-dir -r requirements.txt
    ```

3. **Create a `.env` file in the root directory**:
   ```bash
   touch .env
    ```

   Configure environment variables in a `.env` file (refer to `example.env`).

   ```ini
   OPENAI_API_KEY=sk-xxxxxx
   RAGAS_APP_TOKEN=apt.xxxxxx
   
   MODEL_NAME=gpt-3.5-turbo
   TEMPERATURE=0.0
   MAX_TOKENS=500
   EMBEDDING_MODEL=text-embedding-ada-002
   
   HOST=localhost
   PORT=6333
   ```
---

## Usage

### Local Setup:
1. Launch the chatbot locally:

    ```bash
    python -m src.gradio_ui
     ```
   
   - Access the application at [http://localhost:7860/](http://localhost:7860/).

### Docker Setup:

2. Run the application with Docker to ensure consistent performance:
    ```bash
    docker-compose up --build
    ```
  
   Access via [http://localhost:7860/](http://localhost:7860/).

3. **Interact with the Chatbot**
   Use the interface to upload and index documents, then start interacting by asking specific questions 

    >**Note**: Ensure you have created or selected a QDrant collection before querying.

   - **Indexing a Document**:  
     In the Gradio interface, click **"Index a new PDF"**, upload your document, and specify the name for a new or existing QDrant collection.

   - **Asking Questions**:  
     Navigate to the **"Chatbot"** tab, select your desired collection, and enter your questions. Relevant document chunks will appear in the *Retrieved Chunks* tab.

---

## Enjoy the Chatbot!
![RAG Chatbot Demo Screenshot](./images/img.png "Gradio UI")
![RAG Chatbot Demo Screenshot](./images/img_1.png "Retrieved Chunks")
