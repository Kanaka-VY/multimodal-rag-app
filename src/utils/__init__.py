"""Utility modules: config, text processing, image processing, monitoring."""

__all__ = [
    "get_settings",
    "get_model_config",
    "get_db_config",
    "chunk_text",
    "iter_pdf_pages",
    "load_image",
    "image_caption_stub",
    "INGEST_TOTAL",
    "QUERY_TOTAL",
    "QUERY_LATENCY",
]


def __getattr__(name: str):
    if name == "get_settings":
        from src.utils.config import get_settings

        return get_settings
    if name == "get_model_config":
        from src.utils.config import get_model_config

        return get_model_config
    if name == "get_db_config":
        from src.utils.config import get_db_config

        return get_db_config
    if name == "chunk_text":
        from src.utils.text_processing import chunk_text

        return chunk_text
    if name == "iter_pdf_pages":
        from src.utils.text_processing import iter_pdf_pages

        return iter_pdf_pages
    if name == "load_image":
        from src.utils.image_processing import load_image

        return load_image
    if name == "image_caption_stub":
        from src.utils.image_processing import image_caption_stub

        return image_caption_stub
    if name == "INGEST_TOTAL":
        from src.utils.monitoring import INGEST_TOTAL

        return INGEST_TOTAL
    if name == "QUERY_TOTAL":
        from src.utils.monitoring import QUERY_TOTAL

        return QUERY_TOTAL
    if name == "QUERY_LATENCY":
        from src.utils.monitoring import QUERY_LATENCY

        return QUERY_LATENCY
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
