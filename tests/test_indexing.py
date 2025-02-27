import os
import tempfile
import pytest
from src.indexing import load_documents_from_pdf, index_file_in_qdrant

#Fake QdrantClient to simulate external dependency behavior.
class FakeQdrantClient:
    def __init__(self, *args, **kwargs):
        self.collections = {}

    def get_collection(self, name):
        if name not in self.collections:
            raise Exception("Collection not found")
        return self.collections[name]

    def create_collection(self, collection_name, vectors_config):
        self.collections[collection_name] = {"name": collection_name, "vectors_config": vectors_config}

    def get_collections(self):
        class FakeCollection:
            def __init__(self, name):
                self.name = name
        return type("FakeCollections", (), {"collections": [FakeCollection(name) for name in self.collections]})

#Fixture for overriding QdrantClient in indexing.py with fake client.
@pytest.fixture
def fake_qdrant_client(monkeypatch):
    monkeypatch.setattr("src.indexing.QdrantClient", lambda *args, **kwargs: FakeQdrantClient())

def test_load_documents_from_pdf(temp_pdf_file):
    docs = load_documents_from_pdf(temp_pdf_file)
    #At least one document should be returned.
    assert isinstance(docs, list)
    assert len(docs) >= 1
    for doc in docs:
        assert hasattr(doc, "page_content")
        assert hasattr(doc, "metadata")

def test_index_file_in_qdrant_invalid_file():
    with pytest.raises(ValueError):
        index_file_in_qdrant("nonexistent.pdf", "test_collection")

def test_index_file_in_qdrant_success(temp_pdf_file, fake_qdrant_client):
    #Use the fake Qdrant client via fixture.
    msg = index_file_in_qdrant(temp_pdf_file, "test_collection")
    assert "Successfully indexed" in msg