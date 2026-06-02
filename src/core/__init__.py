"""Core RAG modules: ingest, retriever, llm_handler."""

__all__ = ["IngestPipeline", "MultimodalRetriever", "LLMHandler"]


def __getattr__(name: str):
    if name == "IngestPipeline":
        from src.core.ingest import IngestPipeline

        return IngestPipeline
    if name == "MultimodalRetriever":
        from src.core.retriever import MultimodalRetriever

        return MultimodalRetriever
    if name == "LLMHandler":
        from src.core.llm_handler import LLMHandler

        return LLMHandler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
