import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_text():
    return (
        "Multimodal retrieval augments language models with images, audio, and documents. "
        "Vector databases store embeddings for similarity search."
    )
