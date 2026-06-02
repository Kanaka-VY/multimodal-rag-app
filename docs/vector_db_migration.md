# Vector Database Migration Guide

## Overview

This document provides a migration path from the current ChromaDB implementation to enterprise-grade vector databases for production scaling.

## Current Implementation

- **Vector Database**: ChromaDB (local/embedded)
- **Persistence**: Local file system (`./chroma_data`)
- **Scaling**: Limited to single machine
- **Suitable for**: Development, testing, small-scale deployments (< 1M vectors)

## Migration Options

### Option 1: Pinecone

**Pros:**
- Fully managed, serverless
- Automatic scaling
- Built-in hybrid search (sparse + dense)
- Excellent performance for large-scale deployments
- Easy integration with existing code

**Cons:**
- Cloud-only (no self-hosted option)
- Cost increases with scale
- Vendor lock-in

**Migration Steps:**

1. Install Pinecone client:
```bash
pip install pinecone-client
```

2. Update `configs/db_config.yaml`:
```yaml
vector_db:
  provider: pinecone
  pinecone:
    api_key: ${PINECONE_API_KEY}
    environment: us-east-1-aws
    index_name: omnirag-production
    dimension: 384
    metric: cosine
```

3. Create migration script:
```python
# scripts/migrate_to_pinecone.py
import pinecone
from src.database.vector_store import VectorStore

def migrate_to_pinecone():
    # Initialize Pinecone
    pinecone.init(api_key="your-api-key", environment="us-east-1-aws")
    
    # Create index
    pinecone.create_index(
        name="omnirag-production",
        dimension=384,
        metric="cosine",
        pod_type="p1.x1"
    )
    
    # Migrate data from ChromaDB
    chroma_store = VectorStore()
    all_documents = chroma_store.collection.get()
    
    # Batch upload to Pinecone
    index = pinecone.Index("omnirag-production")
    index.upsert(vectors=all_documents)
```

4. Update `src/database/vector_store.py` to use Pinecone:
```python
class VectorStore:
    def __init__(self):
        cfg = get_db_config()
        provider = cfg.get("vector_db", {}).get("provider", "chroma")
        
        if provider == "pinecone":
            self._init_pinecone(cfg)
        else:
            self._init_chroma(cfg)
```

### Option 2: Qdrant

**Pros:**
- Open source, self-hosted option available
- Excellent performance
- Hybrid search support
- Can be deployed on-premises or cloud
- Active community

**Cons:**
- Requires infrastructure management for self-hosted
- More complex setup than managed services

**Migration Steps:**

1. Install Qdrant client:
```bash
pip install qdrant-client
```

2. Update `configs/db_config.yaml`:
```yaml
vector_db:
  provider: qdrant
  qdrant:
    url: ${QDRANT_URL}
    api_key: ${QDRANT_API_KEY}
    collection_name: omnirag-production
    dimension: 384
```

3. Deploy Qdrant (Docker):
```yaml
# docker-compose.yml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
```

4. Migrate data:
```python
# scripts/migrate_to_qdrant.py
from qdrant_client import QdrantClient
from src.database.vector_store import VectorStore

def migrate_to_qdrant():
    client = QdrantClient(url="http://localhost:6333")
    
    # Create collection
    client.create_collection(
        collection_name="omnirag-production",
        vectors_config={
            "size": 384,
            "distance": "Cosine"
        }
    )
    
    # Migrate data from ChromaDB
    chroma_store = VectorStore()
    all_documents = chroma_store.collection.get()
    
    # Batch upload to Qdrant
    client.upsert(
        collection_name="omnirag-production",
        points=all_documents
    )
```

### Option 3: Milvus

**Pros:**
- Open source, highly scalable
- Supports billions of vectors
- Advanced filtering capabilities
- Cloud-native architecture

**Cons:**
- Complex deployment (requires Kubernetes for production)
- Steeper learning curve
- More infrastructure overhead

**Migration Steps:**

1. Deploy Milvus (using Milvus Operator on Kubernetes):
```yaml
# milvus-deployment.yaml
apiVersion: milvus.io/v1beta1
kind: Milvus
metadata:
  name: my-milvus
spec:
  config:
    components:
      dataNode:
        replicas: 1
      queryNode:
        replicas: 1
      indexNode:
        replicas: 1
```

2. Install PyMilvus:
```bash
pip install pymilvus
```

3. Migrate data:
```python
# scripts/migrate_to_milvus.py
from pymilvus import connections, Collection
from src.database.vector_store import VectorStore

def migrate_to_milvus():
    connections.connect(host="localhost", port="19530")
    
    # Create collection
    collection = Collection("omnirag_production")
    
    # Migrate data from ChromaDB
    chroma_store = VectorStore()
    all_documents = chroma_store.collection.get()
    
    # Batch upload to Milvus
    collection.insert(all_documents)
    collection.flush()
```

## Hybrid Search Implementation

All three options support hybrid search (sparse + dense vectors). Here's how to implement it:

### BM25 + Vector Search

```python
# src/core/hybrid_retriever.py
from rank_bm25 import BM25Okapi
from src.core.retriever import MultimodalRetriever

class HybridRetriever:
    def __init__(self):
        self.vector_retriever = MultimodalRetriever()
        self.bm25_index = None
        self._build_bm25_index()
    
    def _build_bm25_index(self):
        """Build BM25 index from all documents."""
        # Get all documents from vector store
        documents = self._get_all_documents()
        tokenized_docs = [doc["content"].split() for doc in documents]
        self.bm25_index = BM25Okapi(tokenized_docs)
    
    def retrieve(self, query: str, top_k: int = 5):
        """Retrieve using hybrid search."""
        # Sparse search (BM25)
        bm25_scores = self.bm25_index.get_scores(query.split())
        
        # Dense search (vector)
        vector_results = self.vector_retriever.retrieve(query, top_k=top_k * 2)
        
        # Combine scores
        combined_results = self._combine_scores(bm25_scores, vector_results)
        
        # Re-rank with Cohere
        reranked = self._rerank_with_cohere(query, combined_results)
        
        return reranked[:top_k]
```

## Performance Comparison

| Database | Vectors | Latency (p95) | Cost (monthly) | Scaling |
|----------|---------|---------------|----------------|---------|
| ChromaDB | < 1M | 50ms | $0 (self-hosted) | Limited |
| Pinecone | 10M+ | 20ms | $70-500 | Automatic |
| Qdrant | 100M+ | 30ms | $50-300 | Manual/Auto |
| Milvus | 1B+ | 25ms | $100-1000 | Manual |

## Recommendation

**For Production MNC Deployment:**

1. **Start with Pinecone** for fastest time-to-production
2. **Consider Qdrant** if you need self-hosted option or cost control
3. **Use Milvus** only if you need extreme scale (1B+ vectors)

## Migration Checklist

- [ ] Choose target vector database
- [ ] Update configuration files
- [ ] Install required dependencies
- [ ] Deploy target database
- [ ] Create migration script
- [ ] Test migration with sample data
- [ ] Run full migration
- [ ] Verify data integrity
- [ ] Update application code
- [ ] Update CI/CD pipeline
- [ ] Monitor performance
- [ ] Decommission old database

## Rollback Plan

If migration fails:

1. Keep ChromaDB running during migration
2. Use feature flags to switch between databases
3. Maintain backup of ChromaDB data
4. Have rollback script ready to revert changes

```python
# scripts/rollback_to_chroma.py
def rollback_to_chroma():
    """Rollback to ChromaDB if migration fails."""
    cfg = get_db_config()
    cfg["vector_db"]["provider"] = "chroma"
    save_config(cfg)
    print("Rolled back to ChromaDB")
```
