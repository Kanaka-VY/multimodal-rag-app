from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from src.core.ingest import IngestPipeline
from src.core.llm_handler import LLMHandler
from src.core.retriever import MultimodalRetriever
from src.database.vector_store import VectorStore
from src.utils.config import ROOT, get_settings, get_model_config
from src.utils.monitoring import QUERY_LATENCY, QUERY_TOTAL, evaluate_faithfulness

app = FastAPI(
    title="Multimodal RAG API",
    description="Ingest PDF, image, and audio; retrieve and answer with RAG.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = VectorStore()
ingest_pipeline = IngestPipeline(store)
retriever = MultimodalRetriever(store)
llm = LLMHandler()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    modality: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "documents_indexed": store.count()}


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in IngestPipeline.SUPPORTED:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        result = ingest_pipeline.ingest_path(tmp_path)
        result["stored_as"] = file.filename
        return result
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/ingest/directory")
def ingest_directory(path: str = "data") -> dict:
    directory = (ROOT / path).resolve()
    if not directory.is_dir():
        raise HTTPException(404, f"Directory not found: {directory}")
    results = ingest_pipeline.ingest_directory(directory)
    return {"ingested": len(results), "results": results}


@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest) -> QueryResponse:
    start = time.perf_counter()
    try:
        hits = retriever.retrieve(body.query, modality=body.modality, top_k=body.top_k)
        answer = llm.generate(body.query, hits)
        
        # Evaluate faithfulness if enabled
        cfg = get_model_config()
        if cfg.get("observability", {}).get("use_ragas", False):
            faithfulness_score = evaluate_faithfulness(body.query, answer, hits)
            if faithfulness_score is not None and faithfulness_score < 0.5:
                # Low faithfulness detected, add warning
                answer += f"\n\n[Warning: Low faithfulness score ({faithfulness_score:.2f}) - answer may not be well-grounded in retrieved context]"
        
        QUERY_TOTAL.inc()
        return QueryResponse(answer=answer, sources=hits)
    finally:
        QUERY_LATENCY.observe(time.perf_counter() - start)


@app.delete("/collection")
def reset_collection() -> dict:
    store.reset()
    return {"status": "collection reset"}


@app.websocket("/ws/query")
async def websocket_query(websocket: WebSocket) -> None:
    """WebSocket endpoint for streaming query responses."""
    await websocket.accept()
    start = time.perf_counter()
    
    try:
        # Receive query parameters
        data = await websocket.receive_json()
        query = data.get("query", "")
        modality = data.get("modality")
        top_k = data.get("top_k", 5)
        
        if not query:
            await websocket.send_json({"error": "Query is required"})
            await websocket.close()
            return
        
        # Send acknowledgment
        await websocket.send_json({"status": "retrieving"})
        
        # Retrieve documents
        try:
            hits = retriever.retrieve(query, modality=modality, top_k=top_k)
        except Exception as e:
            await websocket.send_json({"error": f"Retrieval failed: {str(e)}"})
            return
        
        # Send retrieved sources
        await websocket.send_json({
            "status": "sources_retrieved",
            "sources": hits
        })
        
        # Send generation status
        await websocket.send_json({"status": "generating"})
        
        # Generate answer (streaming simulation)
        try:
            answer = llm.generate(query, hits)
        except Exception as e:
            await websocket.send_json({"error": f"Generation failed: {str(e)}"})
            return
        
        # Stream the answer word by word
        words = answer.split()
        for i, word in enumerate(words):
            chunk = word + " " if i < len(words) - 1 else word
            await websocket.send_json({
                "status": "streaming",
                "chunk": chunk,
                "progress": (i + 1) / len(words)
            })
            # Small delay to simulate streaming
            import asyncio
            await asyncio.sleep(0.02)
        
        # Send completion status
        await websocket.send_json({
            "status": "complete",
            "answer": answer,
            "sources": hits
        })
        
        QUERY_TOTAL.inc()
        
    except WebSocketDisconnect:
        pass
    except Exception as e:
        import traceback
        error_details = f"{str(e)}\n{traceback.format_exc()}"
        await websocket.send_json({"error": f"Internal error: {error_details}"})
    finally:
        QUERY_LATENCY.observe(time.perf_counter() - start)
        try:
            await websocket.close()
        except:
            pass


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    run()
