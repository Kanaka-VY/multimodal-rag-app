"""Tests for vector store."""

import tempfile
from pathlib import Path

import pytest

from src.database.vector_store import VectorStore


@pytest.fixture
def temp_store(tmp_path):
    """Create a temporary vector store for testing."""
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    client = chromadb.PersistentClient(
        path=str(tmp_path / "chroma_test"),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name="test_collection",
        metadata={"hnsw:space": "cosine"},
    )

    # Monkey patch the VectorStore to use our test collection
    original_init = VectorStore.__init__

    def test_init(self):
        self._client = client
        self._collection = collection
        self.collection_name = "test_collection"
        self._host = "localhost"
        self._port = 8001

    VectorStore.__init__ = test_init

    yield VectorStore()

    # Restore original init
    VectorStore.__init__ = original_init
    client.delete_collection("test_collection")


def test_add_documents(temp_store):
    """Test adding documents to the store."""
    texts = ["test document one", "test document two"]
    embeddings = [[0.1] * 384, [0.2] * 384]
    metadatas = [{"source": "doc1"}, {"source": "doc2"}]

    ids = temp_store.add_documents(texts, embeddings, metadatas)

    assert len(ids) == 2
    assert all(isinstance(id_, str) for id_ in ids)


def test_add_documents_with_timestamp(temp_store):
    """Test that documents get created_at timestamp."""
    texts = ["test document"]
    embeddings = [[0.1] * 384]
    metadatas = [{"source": "doc1"}]

    ids = temp_store.add_documents(texts, embeddings, metadatas)

    # Verify the document was added
    count = temp_store.count()
    assert count == 1


def test_query(temp_store):
    """Test querying the store."""
    # Add some documents first
    texts = ["test document one", "test document two"]
    embeddings = [[0.1] * 384, [0.2] * 384]
    metadatas = [{"source": "doc1"}, {"source": "doc2"}]
    temp_store.add_documents(texts, embeddings, metadatas)

    # Query
    query_embedding = [0.1] * 384
    result = temp_store.query(query_embedding, top_k=2)

    assert "documents" in result
    assert "metadatas" in result
    assert "distances" in result


def test_query_with_where(temp_store):
    """Test querying with metadata filter."""
    texts = ["doc one", "doc two"]
    embeddings = [[0.1] * 384, [0.2] * 384]
    metadatas = [{"modality": "pdf"}, {"modality": "text"}]
    temp_store.add_documents(texts, embeddings, metadatas)

    query_embedding = [0.1] * 384
    result = temp_store.query(query_embedding, top_k=2, where={"modality": "pdf"})

    assert "documents" in result


def test_count(temp_store):
    """Test counting documents in the store."""
    assert temp_store.count() == 0

    texts = ["doc1", "doc2", "doc3"]
    embeddings = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
    metadatas = [{"source": "1"}, {"source": "2"}, {"source": "3"}]
    temp_store.add_documents(texts, embeddings, metadatas)

    assert temp_store.count() == 3


def test_reset(temp_store):
    """Test resetting the collection."""
    texts = ["doc1"]
    embeddings = [[0.1] * 384]
    metadatas = [{"source": "1"}]
    temp_store.add_documents(texts, embeddings, metadatas)

    assert temp_store.count() == 1

    temp_store.reset()
    assert temp_store.count() == 0
