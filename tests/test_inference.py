import pytest


# Fake retriever and vectorstore to simulate Qdrant-based retrieval.
class FakeRetriever:
    def get_relevant_documents(self, query):
        class FakeDoc:
            def __init__(self, content, page):
                self.page_content = content
                self.metadata = {"page": page}

        return [FakeDoc("Fake context from page 1.", 1)]


class FakeVectorStore:
    def as_retriever(self, search_kwargs):
        return FakeRetriever()


def fake_vectorstore_init(client, collection_name, embeddings):
    return FakeVectorStore()


# Fake ChatOpenAI that returns a known answer.
class FakeChatOpenAI:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, prompt):
        return "Fake LLM answer"


def fake_chat_openai_init(model_name, temperature, max_tokens, openai_api_key):
    return FakeChatOpenAI()


def test_answer_query_no_collection(monkeypatch):
    from src.inference import answer_query
    answer, context = answer_query("What is revenue?", history=[], collection_name="")
    assert "No Qdrant collection selected" in answer


def test_answer_query_success(monkeypatch):
    # Override Qdrant initialization and ChatOpenAI constructor.
    monkeypatch.setattr("src.inference.Qdrant", fake_vectorstore_init)
    monkeypatch.setattr("src.inference.ChatOpenAI", fake_chat_openai_init)

    from src.inference import answer_query
    answer, context = answer_query(
        "Test query",
        history=[{"role": "user", "content": "Test query"}],
        collection_name="test_collection"
    )
    assert "Fake LLM answer" in answer
    assert "Page 1" in context  # Checking for merged context that includes page details.
