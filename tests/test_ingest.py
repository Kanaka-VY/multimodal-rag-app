"""Tests for ingest pipeline."""

import tempfile
from pathlib import Path

import pytest
from PIL import Image

from src.core.ingest import IngestPipeline
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


@pytest.fixture
def pipeline(temp_store):
    """Create ingest pipeline with test store."""
    return IngestPipeline(temp_store)


def test_ingest_text_file(pipeline, tmp_path):
    """Test ingesting a text file."""
    text_file = tmp_path / "test.txt"
    text_file.write_text("This is a test document with some content.")

    result = pipeline.ingest_path(text_file)

    assert result["filename"] == "test.txt"
    assert result["modality"] == "text"
    assert result["chunks"] > 0
    assert "ids" in result


def test_ingest_pdf(pipeline, tmp_path):
    """Test ingesting a PDF file."""
    from pypdf import PdfWriter

    pdf_file = tmp_path / "test.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(pdf_file, "wb") as f:
        writer.write(f)

    result = pipeline.ingest_path(pdf_file)

    assert result["filename"] == "test.pdf"
    assert result["modality"] == "pdf"


def test_ingest_image(pipeline, tmp_path):
    """Test ingesting an image file."""
    img_file = tmp_path / "test.png"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_file)

    result = pipeline.ingest_path(img_file)

    assert result["filename"] == "test.png"
    assert result["modality"] == "image"
    assert result["chunks"] == 1


def test_ingest_unsupported_file(pipeline, tmp_path):
    """Test ingesting an unsupported file type raises error."""
    unsupported = tmp_path / "test.xyz"
    unsupported.write_text("content")

    with pytest.raises(ValueError, match="Unsupported file type"):
        pipeline.ingest_path(unsupported)


def test_ingest_directory(pipeline, tmp_path):
    """Test ingesting a directory with multiple files."""
    # Create test files
    (tmp_path / "doc1.txt").write_text("Document one content")
    (tmp_path / "doc2.txt").write_text("Document two content")

    results = pipeline.ingest_directory(tmp_path)

    assert len(results) == 2
    assert all(r["modality"] == "text" for r in results)


def test_supported_extensions():
    """Test that supported extensions include expected types."""
    expected = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".wav", ".mp3", ".m4a", ".txt"}
    assert IngestPipeline.SUPPORTED == expected
