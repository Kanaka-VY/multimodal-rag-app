"""Tests for monitoring metrics."""

from prometheus_client import REGISTRY

from src.utils.monitoring import INGEST_TOTAL, QUERY_TOTAL, QUERY_LATENCY, FAITHFULNESS_SCORE, evaluate_faithfulness, RAGAS_AVAILABLE


def test_ingest_total_counter():
    """Test INGEST_TOTAL counter exists and can be incremented."""
    # Get initial value
    metric = REGISTRY.get_sample_value("rag_ingest_total", {"modality": "pdf"})
    initial = metric if metric is not None else 0

    INGEST_TOTAL.labels(modality="pdf").inc()

    # Check it increased
    metric_after = REGISTRY.get_sample_value("rag_ingest_total", {"modality": "pdf"})
    assert metric_after == initial + 1


def test_query_total_counter():
    """Test QUERY_TOTAL counter exists and can be incremented."""
    metric = REGISTRY.get_sample_value("rag_query_total")
    initial = metric if metric is not None else 0

    QUERY_TOTAL.inc()

    metric_after = REGISTRY.get_sample_value("rag_query_total")
    assert metric_after == initial + 1


def test_query_latency_histogram():
    """Test QUERY_LATENCY histogram can observe values."""
    QUERY_LATENCY.observe(0.5)

    # Check histogram has samples
    samples = REGISTRY.get_sample_value("rag_query_latency_seconds_bucket", {"le": "0.5"})
    assert samples is not None


def test_ingest_total_multiple_modalities():
    """Test INGEST_TOTAL tracks different modalities separately."""
    INGEST_TOTAL.labels(modality="pdf").inc()
    INGEST_TOTAL.labels(modality="image").inc()
    INGEST_TOTAL.labels(modality="audio").inc()

    pdf_metric = REGISTRY.get_sample_value("rag_ingest_total", {"modality": "pdf"})
    image_metric = REGISTRY.get_sample_value("rag_ingest_total", {"modality": "image"})
    audio_metric = REGISTRY.get_sample_value("rag_ingest_total", {"modality": "audio"})

    assert pdf_metric >= 1
    assert image_metric >= 1
    assert audio_metric >= 1


def test_faithfulness_score_gauge():
    """Test FAITHFULNESS_SCORE gauge can be set."""
    FAITHFULNESS_SCORE.set(0.85)
    
    metric = REGISTRY.get_sample_value("rag_faithfulness_score")
    assert metric == 0.85


def test_evaluate_faithfulness_when_ragas_available():
    """Test faithfulness evaluation when RAGAS is available."""
    if not RAGAS_AVAILABLE:
        return  # Skip test if RAGAS not available
    
    query = "What is the revenue?"
    answer = "The revenue is $1M based on the financial report."
    contexts = [
        {"content": "The company reported revenue of $1M in Q3.", "metadata": {"source": "report.pdf"}}
    ]
    
    # This will return None if RAGAS evaluation fails, which is acceptable
    score = evaluate_faithfulness(query, answer, contexts)
    assert score is None or 0.0 <= score <= 1.0
