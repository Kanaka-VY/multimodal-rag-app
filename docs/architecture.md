# Architecture

## Components

1. **IngestPipeline** — Normalizes PDF, image, and audio into text or embeddings, then writes to ChromaDB.
2. **MultimodalRetriever** — Runs text and/or CLIP query embeddings; merges cross-modal hits.
3. **LLMHandler** — Local template answers or OpenAI chat completion over retrieved context.
4. **VectorStore** — Persistent Chroma collection with cosine similarity.

## Data flow

1. Raw files land in `data/`.
2. `migrate_data.py` or `/ingest/file` runs the pipeline.
3. User queries via Streamlit → `POST /query`.
4. Retriever returns top-k chunks; LLM synthesizes an answer with citations.

## Observability

- Prometheus counters: `rag_ingest_total`, `rag_query_total`
- Histogram: `rag_query_latency_seconds`
- Scraped from `/metrics` on the API service
