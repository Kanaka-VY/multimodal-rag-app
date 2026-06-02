from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from src.utils.config import get_model_config, get_settings

if TYPE_CHECKING:
    from PIL import Image
    from sentence_transformers import SentenceTransformer


@lru_cache
def _text_model() -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    cfg = get_model_config()
    model_name = cfg.get("embedding", {}).get(
        "text_model",
        get_settings().embedding_model,
    )
    return SentenceTransformer(model_name)


@lru_cache
def _clip_model() -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    cfg = get_model_config()
    model_name = cfg.get("embedding", {}).get(
        "clip_model",
        get_settings().clip_model,
    )
    return SentenceTransformer(model_name)


def embed_text(texts: list[str]) -> list[list[float]]:
    model = _text_model()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def embed_text_query(query: str) -> list[float]:
    return embed_text([query])[0]


def embed_image(image: Image.Image) -> list[float]:
    model = _clip_model()
    vector = model.encode(image, convert_to_numpy=True, show_progress_bar=False)
    return vector.tolist()


def embed_image_query(query: str) -> list[float]:
    """CLIP can encode text queries for image-style retrieval."""
    model = _clip_model()
    vector = model.encode(query, convert_to_numpy=True, show_progress_bar=False)
    return vector.tolist()
