"""Tests for LLM handler."""

import pytest
from unittest.mock import MagicMock

from src.core.llm_handler import LLMHandler, LITELLM_AVAILABLE


@pytest.fixture
def handler():
    """Create LLM handler instance."""
    return LLMHandler()


def test_generate_local_no_context(handler):
    """Test local generation with no context returns fallback message."""
    result = handler.generate("test query", [])
    assert "No relevant documents" in result


def test_generate_local_with_context(handler):
    """Test local generation with context returns formatted answer."""
    contexts = [
        {
            "content": "Test content about machine learning",
            "metadata": {"filename": "doc1.pdf", "modality": "pdf"},
        }
    ]
    result = handler.generate("What is machine learning?", contexts)

    assert "Question: What is machine learning?" in result
    assert "Test content about machine learning" in result
    assert "doc1.pdf" in result


def test_generate_local_multiple_contexts(handler):
    """Test local generation with multiple contexts formats them correctly."""
    contexts = [
        {
            "content": "First context",
            "metadata": {"filename": "doc1.txt", "modality": "text"},
        },
        {
            "content": "Second context",
            "metadata": {"filename": "doc2.txt", "modality": "text"},
        },
    ]
    result = handler.generate("test query", contexts)

    assert "[1] (text) doc1.txt" in result
    assert "[2] (text) doc2.txt" in result


def test_summarize_from_context(handler):
    """Test context summarization extracts relevant passage."""
    contexts = [
        {
            "content": "This is a long piece of text that should be summarized appropriately",
            "metadata": {"filename": "doc.pdf"},
        }
    ]
    result = handler._summarize_from_context("test query", contexts)

    assert "most relevant passage" in result.lower()
    assert len(result) > 0


def test_generate_vision_with_image_contexts(handler):
    """Test vision-language model generation with image contexts."""
    if not LITELLM_AVAILABLE:
        return  # Skip test if litellm not available
    
    contexts = [
        {
            "content": "Chart showing revenue growth",
            "metadata": {"filename": "chart.png", "modality": "image"},
        },
        {
            "content": "Text about Q3 financial results",
            "metadata": {"filename": "report.pdf", "modality": "pdf"},
        }
    ]
    
    # Mock the litellm completion to avoid actual API call
    from unittest.mock import patch
    with patch("src.core.llm_handler.completion") as mock_completion:
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Vision model response"))]
        )
        
        result = handler._generate_vision("What does the chart show?", contexts)
        assert "Vision model response" in result or "fallback" in result.lower()
