"""Tests for embedding functions."""

import pytest
from PIL import Image

from src.core.embeddings import embed_text, embed_text_query, embed_image, embed_image_query


def test_embed_text():
    """Test text embedding returns correct dimensions."""
    texts = ["test sentence one", "test sentence two"]
    embeddings = embed_text(texts)

    assert len(embeddings) == 2
    assert all(isinstance(emb, list) for emb in embeddings)
    assert all(len(emb) == 384 for emb in embeddings)


def test_embed_text_query():
    """Test text query embedding returns correct dimension."""
    query = "test query"
    embedding = embed_text_query(query)

    assert isinstance(embedding, list)
    assert len(embedding) == 384


def test_embed_image(tmp_path):
    """Test image embedding returns correct dimension."""
    # Create a simple test image
    img = Image.new("RGB", (100, 100), color="red")
    embedding = embed_image(img)

    assert isinstance(embedding, list)
    assert len(embedding) == 512


def test_embed_image_query():
    """Test image query embedding returns correct dimension."""
    query = "a red square"
    embedding = embed_image_query(query)

    assert isinstance(embedding, list)
    assert len(embedding) == 512


def test_embed_text_empty_list():
    """Test embedding empty list returns empty list."""
    embeddings = embed_text([])
    assert embeddings == []
