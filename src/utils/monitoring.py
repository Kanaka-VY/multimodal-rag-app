from prometheus_client import Counter, Histogram, Gauge

INGEST_TOTAL = Counter(
    "rag_ingest_total",
    "Documents ingested",
    ["modality"],
)
QUERY_TOTAL = Counter("rag_query_total", "RAG queries executed")
QUERY_LATENCY = Histogram(
    "rag_query_latency_seconds",
    "RAG query latency",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
FAITHFULNESS_SCORE = Gauge(
    "rag_faithfulness_score",
    "RAG answer faithfulness score",
)

try:
    from ragas import evaluate
    from ragas.metrics import faithfulness
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

try:
    from phoenix.otel import register
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False

# Phoenix tracer
_phoenix_tracer = None

def initialize_phoenix(endpoint: str = "http://localhost:6006", project_name: str = "omnirag-multimodal") -> None:
    """Initialize Arize Phoenix for trace inspection."""
    global _phoenix_tracer
    
    if not PHOENIX_AVAILABLE:
        return
    
    try:
        register(
            project_name=project_name,
            endpoint=endpoint,
        )
        
        # Set up tracer
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(register))
        trace.set_tracer_provider(provider)
        
        _phoenix_tracer = trace.get_tracer(__name__)
    except Exception:
        _phoenix_tracer = None

def get_phoenix_tracer():
    """Get the Phoenix tracer instance."""
    return _phoenix_tracer


def evaluate_faithfulness(query: str, answer: str, contexts: list[dict]) -> float | None:
    """Evaluate faithfulness of RAG answer using RAGAS."""
    if not RAGAS_AVAILABLE:
        return None
    
    try:
        # Prepare data for RAGAS evaluation
        context_texts = [ctx.get("content", "") for ctx in contexts]
        
        dataset = Dataset.from_dict({
            "question": [query],
            "answer": [answer],
            "contexts": [context_texts],
        })
        
        # Evaluate faithfulness
        result = evaluate(
            dataset,
            metrics=[faithfulness],
        )
        
        # Extract faithfulness score
        score = result["faithfulness"][0] if "faithfulness" in result else 0.0
        
        # Update Prometheus gauge
        FAITHFULNESS_SCORE.set(score)
        
        return score
    except Exception:
        # Evaluation failed, return None
        return None
