from collections import namedtuple
import os
import pytest

from src import gradio_ui, indexing, inference

class FakeCollection:
    def __init__(self, name):
        self.name = name

class FakeQdrantClient:
    def __init__(self, *args, **kwargs):
        pass

    def get_collections(self):
        class FakeCollections:
            collections = [FakeCollection("test_collection_1"), FakeCollection("test_collection_2")]
        return FakeCollections()

    def create_collection(self, collection_name, vectors_config):
        # Simulate successful creation.
        pass

# Fake file object with a .name attribute (simulate a PDF file)
class FakeFile:
    def __init__(self, name):
        self.name = name

# fake document for testing merge_chunks.
FakeDoc = lambda content, page: type("FakeDoc", (object,), {"page_content": content, "metadata": {"page": page}})

# ------------------------------------------------------------------
# Tests for gradio_ui.py
def test_list_qdrant_collections(monkeypatch):
    # Monkeypatch QdrantClient to use our fake client.
    monkeypatch.setattr(gradio_ui, "QdrantClient", lambda **kwargs: FakeQdrantClient())
    collections = gradio_ui.list_qdrant_collections()
    assert collections == ["test_collection_1", "test_collection_2"]

def test_on_refresh_collections(monkeypatch):
    # Replace list_qdrant_collections to return a test list.
    monkeypatch.setattr(gradio_ui, "list_qdrant_collections", lambda: ["col1", "col2"])
    update = gradio_ui.on_refresh_collections()
    # We expect the update to contain choices with the list and set value to the first item.
    assert update is not None

def test_perform_index_no_pdf():
    # When no PDF is provided, perform_index should return an error message.
    msg, _ = gradio_ui.perform_index(None, "collection")
    assert msg == "No PDF selected."

def test_perform_index(monkeypatch):
    fake_file = FakeFile("dummy.pdf")

    def fake_index(pdf_path, coll_name):
        return f"Indexed {pdf_path} into {coll_name}"

    monkeypatch.setattr(gradio_ui, "index_file_in_qdrant", fake_index)
    monkeyatch_collections = lambda: ["test_collection"]
    monkeypatch.setattr(gradio_ui, "list_qdrant_collections", monkeyatch_collections)
    msg, _ = gradio_ui.perform_index(fake_file, "test_collection")
    assert "Successfully" in msg

def test_chat_fn_no_collection():
    # chat_fn should prompt for collection selection if none is provided.
    ans, retrieved = gradio_ui.chat_fn("test query", [], "")
    assert ans == "Please select a collection."
    assert retrieved == ""

def test_chat_fn(monkeypatch):
    fake_response = ("Fake answer", "Fake retrieved text")
    monkeypatch.setattr(gradio_ui, "answer_query", lambda query, history, coll: fake_response)
    ans, retrieved = gradio_ui.chat_fn("test query", [{"role": "user", "content": "hi"}], "test_collection")
    assert ans == "Fake answer"
    assert retrieved == "Fake retrieved text"

def test_respond(monkeypatch):
    # Test that respond updates the conversation history correctly.
    fake_response = ("Fake answer", "Fake retrieved text")
    monkeypatch.setattr(gradio_ui, "chat_fn", lambda query, history, coll: fake_response)
    initial_history = []
    updated_history, new_history, retrieved = gradio_ui.respond("Hello", initial_history, "test_collection")
    assert len(updated_history) == 2
    assert updated_history[0]["role"] == "user"
    assert "Hello" in updated_history[0]["content"]

# Tests for indexing.py
def test_sentence_window_text_splitter():
    # Create a splitter with a chunk size of 2 and overlap of 1.
    splitter = indexing.SentenceWindowTextSplitter(chunk_size=2, chunk_overlap=1)
    text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
    chunks = splitter.split_text(text)
    expected_chunks = [
        "This is sentence one. This is sentence two.",
        "This is sentence two. This is sentence three.",
        "This is sentence three. This is sentence four.",
        "This is sentence four."
    ]
    assert chunks == expected_chunks

def test_split_documents(monkeypatch):
    FakeDocument = namedtuple("FakeDocument", ["page_content", "metadata"])
    doc = FakeDocument(page_content="Sentence one. Sentence two. Sentence three.", metadata={"page": 1})
    splitter = indexing.SentenceWindowTextSplitter(chunk_size=2, chunk_overlap=1)
    docs = splitter.split_documents([doc])
    # We expect two or more documents based on the splitting logic.
    assert len(docs) >= 2
    for new_doc in docs:
        # Check that metadata contains both original page and chunk_index
        assert "page" in new_doc.metadata
        assert "chunk_index" in new_doc.metadata
        assert new_doc.metadata["page"] == 1
        assert isinstance(new_doc.metadata["chunk_index"], int)

# Additional Tests for SentenceWindowTextSplitter
def test_sentence_window_text_splitter_edge_cases():
    splitter = indexing.SentenceWindowTextSplitter(chunk_size=2, chunk_overlap=1)

    # Test empty text
    assert splitter.split_text("") == []

    # Test single sentence
    single = "This is one sentence."
    assert splitter.split_text(single) == [single]

    # Test with invalid chunk_size and overlap
    with pytest.raises(ValueError):
        indexing.SentenceWindowTextSplitter(chunk_size=2, chunk_overlap=2)

    # Test with very long sentences
    long_text = "This is a very long sentence that should still be processed. " * 5
    chunks = splitter.split_text(long_text)
    assert len(chunks) > 0

# Tests for PDF Processing
def test_load_documents_from_pdf_empty(temp_pdf_file):
    # Test with empty PDF
    docs = indexing.load_documents_from_pdf(temp_pdf_file)
    assert isinstance(docs, list)
    assert len(docs) > 0
    assert all(hasattr(doc, 'page_content') for doc in docs)
    assert all(hasattr(doc, 'metadata') for doc in docs)

def test_load_documents_from_pdf_invalid():
    # Test with non-existent file
    with pytest.raises(Exception):
        indexing.load_documents_from_pdf("nonexistent.pdf")


# Tests for QDrant Collection Management
def test_create_qdrant_collection_invalid_params(monkeypatch):
    class MockQdrantClient:
        def create_collection(self, collection_name, vectors_config):
            if not collection_name:
                raise ValueError("Invalid collection name")
            if vectors_config.size <= 0:
                raise ValueError("Invalid vector size")
            return True

    monkeypatch.setattr(indexing, "QdrantClient", lambda host, port: MockQdrantClient())

    # Test with empty collection name
    with pytest.raises(ValueError):
        indexing.create_qdrant_collection(MockQdrantClient(), "", vector_size=1536)

    # Test with invalid vector size
    with pytest.raises(ValueError):
        indexing.create_qdrant_collection(MockQdrantClient(), "test", vector_size=-1)

# Test for Chat Functions
def test_chat_fn_empty_query():
    # Test with empty query
    ans, retrieved = gradio_ui.chat_fn("", [], "test_collection")
    assert "empty" in ans.lower() or "invalid" in ans.lower()
    assert retrieved == ""

def test_chat_fn_long_history(monkeypatch):
    # Test with a very long conversation history
    fake_response = ("Answer", "Context")
    monkeypatch.setattr(gradio_ui, "answer_query", lambda query, history, coll: fake_response)

    long_history = [{"role": "user", "content": f"Message {i}"} for i in range(50)]
    ans, retrieved = gradio_ui.chat_fn("test", long_history, "test_collection")
    assert ans == "Answer"
    assert retrieved == "Context"

# Tests for inference.py
def test_merge_chunks():
    # Create fake documents to test merge_chunks.
    docs = [
        FakeDoc("Text1", 1),
        FakeDoc("Text2", 1),
        FakeDoc("Text3", 2)
    ]
    merged = inference.merge_chunks(docs)
    assert "(Page 1)" in merged
    assert "Text1" in merged
    assert "Text2" in merged
    assert "(Page 2)" in merged
    assert "Text3" in merged

def test_answer_query(monkeypatch):
    # For answer_query, we need to simulate:
    # - The retrieval of relevant documents.
    # - The LLM response.
    class FakeRetriever:
        def get_relevant_documents(self, query):
            return [FakeDoc("Fake context text.", 1)]

    class FakeVectorStore:
        def as_retriever(self, search_kwargs):
            return FakeRetriever()

    class FakeLLM:
        def predict(self, prompt):
            return "Fake LLM answer"

    monkeypatch.setattr(inference, "OpenAIEmbeddings", lambda model, openai_api_key: None)
    monkeypatch.setattr(inference, "QdrantClient", lambda host, port: None)
    monkeypatch.setattr(inference, "Qdrant", lambda client, collection_name, embeddings: FakeVectorStore())
    monkeypatch.setattr(inference, "ChatOpenAI", lambda model_name, temperature, max_tokens, openai_api_key: FakeLLM())
    answer, merged_content = inference.answer_query("test question", [{"role": "user", "content": "hi"}],
                                                    "test_collection")
    assert "Fake LLM answer" in answer
    assert "Fake context text." in merged_content

# Test for Inference Module
def test_merge_chunks_edge_cases():
    assert inference.merge_chunks([]) == ""

    # Test with single document
    single_doc = FakeDoc("Single text", 1)
    merged = inference.merge_chunks([single_doc])
    assert "Single text" in merged
    assert "(Page 1)" in merged

    # Test with documents having same page number
    same_page_docs = [
        FakeDoc("First part.", 1),
        FakeDoc("First part. Second part.", 1),
        FakeDoc("Second part. Third part.", 1)
    ]
    merged = inference.merge_chunks(same_page_docs)
    assert "First part" in merged
    assert "Second part" in merged
    assert "Third part" in merged
    assert merged.count("First part") == 1  # Check deduplication

def test_answer_query_invalid_inputs():
    # Test with empty inputs
    with pytest.raises(ValueError):
        inference.answer_query("", [], "")

    # Test with None values
    with pytest.raises(ValueError):
        inference.answer_query(None, None, None)

# Test environment variable handling
def test_environment_variables():
    # Test that required environment variables are loaded
    assert os.getenv("OPENAI_API_KEY") is not None
    assert os.getenv("MODEL_NAME") is not None
    assert os.getenv("TEMPERATURE") is not None
    assert os.getenv("MAX_TOKENS") is not None
    assert os.getenv("EMBEDDING_MODEL") is not None
    assert os.getenv("PORT") is not None
    assert os.getenv("HOST") is not None