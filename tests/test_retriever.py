from unittest.mock import MagicMock, patch

from src.core.retriever import MultimodalRetriever, BM25_AVAILABLE, COHERE_AVAILABLE


def test_parse_hits_filters_by_threshold():
    retriever = MultimodalRetriever(store=MagicMock())
    raw = {
        "documents": [["good doc", "weak doc"]],
        "metadatas": [[{"modality": "text"}, {"modality": "text"}]],
        "distances": [[0.1, 0.95]],
    }
    hits = retriever._parse_hits(raw, threshold=0.3)
    assert len(hits) == 1
    assert hits[0]["content"] == "good doc"
    assert hits[0]["score"] > 0.8


@patch("src.core.retriever.embed_text_query", return_value=[0.1] * 8)
@patch("src.core.retriever.embed_image_query", return_value=[0.2] * 8)
def test_retrieve_merges_modalities(mock_clip, mock_text):
    store = MagicMock()
    store.query.side_effect = [
        {
            "documents": [["pdf chunk"]],
            "metadatas": [[{"modality": "pdf", "filename": "a.pdf"}]],
            "distances": [[0.2]],
        },
        {
            "documents": [["image caption"]],
            "metadatas": [[{"modality": "image", "filename": "b.png"}]],
            "distances": [[0.15]],
        },
    ]
    retriever = MultimodalRetriever(store=store)
    hits = retriever.retrieve("test query", top_k=3)
    assert len(hits) <= 3
    assert any(h["metadata"]["modality"] == "pdf" for h in hits)


def test_hybrid_retrieve_when_bm25_available():
    """Test hybrid retrieval when BM25 is available."""
    if not BM25_AVAILABLE:
        return  # Skip test if BM25 not available
    
    store = MagicMock()
    store._collection.get.return_value = {
        "documents": ["doc1 test", "doc2 example", "doc3 sample"],
        "metadatas": [{"modality": "text"}, {"modality": "text"}, {"modality": "text"}],
    }
    store.query.side_effect = [
        {
            "documents": [["doc1 test"]],
            "metadatas": [[{"modality": "text"}]],
            "distances": [[0.2]],
        },
    ]
    
    retriever = MultimodalRetriever(store=store)
    retriever._cfg = {"use_hybrid": True, "top_k": 5, "score_threshold": 0.3}
    retriever._initialize_bm25()
    
    hits = retriever.retrieve("test query", top_k=3)
    assert isinstance(hits, list)


def test_rerank_when_cohere_available():
    """Test re-ranking when Cohere is available."""
    if not COHERE_AVAILABLE:
        return  # Skip test if Cohere not available
    
    retriever = MultimodalRetriever(store=MagicMock())
    retriever._cohere_client = MagicMock()
    retriever._cohere_client.rerank.return_value = MagicMock(
        results=[
            MagicMock(index=1, relevance_score=0.9),
            MagicMock(index=0, relevance_score=0.7),
        ]
    )
    retriever._cfg = {"use_rerank": True, "rerank_top_k": 10}
    
    results = [
        {"content": "doc1", "metadata": {"modality": "text"}, "score": 0.5},
        {"content": "doc2", "metadata": {"modality": "text"}, "score": 0.6},
    ]
    
    reranked = retriever._rerank("test query", results, top_k=2)
    assert len(reranked) == 2
    assert all("reranked" in r for r in reranked)
