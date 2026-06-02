"""
Hybrid Search Retriever
Combines BM25 sparse search with dense vector search for improved retrieval.
"""

from typing import Any

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

from src.core.retriever import MultimodalRetriever
from src.database.vector_store import VectorStore
from src.utils.config import get_model_config


class HybridRetriever:
    """
    Hybrid retriever combining BM25 sparse search with dense vector search.
    """
    
    def __init__(self):
        self.vector_retriever = MultimodalRetriever()
        self.bm25_index = None
        self.documents = []
        self._build_bm25_index()
        
        cfg = get_model_config()
        self.bm25_weight = cfg.get("retrieval", {}).get("bm25_weight", 0.3)
        self.vector_weight = cfg.get("retrieval", {}).get("vector_weight", 0.7)
    
    def _build_bm25_index(self) -> None:
        """Build BM25 index from all documents in vector store."""
        if not BM25_AVAILABLE:
            return
        
        try:
            store = VectorStore()
            all_docs = store.collection.get()
            
            if not all_docs or not all_docs.get("documents"):
                return
            
            self.documents = []
            tokenized_docs = []
            
            for i, doc in enumerate(all_docs["documents"]):
                self.documents.append({
                    "id": all_docs["ids"][i],
                    "content": doc,
                    "metadata": all_docs["metadatas"][i] if all_docs.get("metadatas") else {}
                })
                tokenized_docs.append(doc.split())
            
            self.bm25_index = BM25Okapi(tokenized_docs)
            
        except Exception:
            self.bm25_index = None
    
    def _bm25_search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Perform BM25 sparse search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of documents with BM25 scores
        """
        if not self.bm25_index:
            return []
        
        tokenized_query = query.split()
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top-k results
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "document": self.documents[idx],
                    "score": scores[idx],
                    "method": "bm25"
                })
        
        return results
    
    def _vector_search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Perform dense vector search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of documents with vector similarity scores
        """
        results = self.vector_retriever.retrieve(query, top_k=top_k, modality=None)
        
        return [
            {
                "document": {
                    "id": result.get("id", ""),
                    "content": result.get("content", ""),
                    "metadata": result.get("metadata", {})
                },
                "score": result.get("score", 0.0),
                "method": "vector"
            }
            for result in results
        ]
    
    def _combine_scores(self, bm25_results: list[dict], vector_results: list[dict]) -> list[dict]:
        """
        Combine BM25 and vector search results using weighted scoring.
        
        Args:
            bm25_results: Results from BM25 search
            vector_results: Results from vector search
            
        Returns:
            Combined results with hybrid scores
        """
        # Normalize scores
        bm25_max = max([r["score"] for r in bm25_results], default=1.0)
        vector_max = max([r["score"] for r in vector_results], default=1.0)
        
        # Create document ID to result mapping
        combined = {}
        
        # Add BM25 results
        for result in bm25_results:
            doc_id = result["document"]["id"]
            normalized_score = result["score"] / bm25_max if bm25_max > 0 else 0
            combined[doc_id] = {
                "document": result["document"],
                "bm25_score": normalized_score,
                "vector_score": 0.0,
                "hybrid_score": normalized_score * self.bm25_weight
            }
        
        # Add vector results and combine
        for result in vector_results:
            doc_id = result["document"]["id"]
            normalized_score = result["score"] / vector_max if vector_max > 0 else 0
            
            if doc_id in combined:
                combined[doc_id]["vector_score"] = normalized_score
                combined[doc_id]["hybrid_score"] += normalized_score * self.vector_weight
            else:
                combined[doc_id] = {
                    "document": result["document"],
                    "bm25_score": 0.0,
                    "vector_score": normalized_score,
                    "hybrid_score": normalized_score * self.vector_weight
                }
        
        # Sort by hybrid score
        sorted_results = sorted(combined.values(), key=lambda x: x["hybrid_score"], reverse=True)
        
        return sorted_results
    
    def _rerank_with_cohere(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        """
        Re-rank documents using Cohere re-ranker.
        
        Args:
            query: Search query
            documents: List of documents to re-rank
            top_k: Number of top results to return
            
        Returns:
            Re-ranked documents
        """
        cfg = get_model_config()
        use_rerank = cfg.get("retrieval", {}).get("use_rerank", False)
        
        if not use_rerank:
            return documents[:top_k]
        
        try:
            import cohere
            
            api_key = cfg.get("llm", {}).get("cohere_api_key")
            if not api_key:
                return documents[:top_k]
            
            co = cohere.Client(api_key=api_key)
            
            # Prepare documents for re-ranking
            docs = [doc["document"]["content"] for doc in documents]
            
            # Re-rank
            rerank_results = co.rerank(
                query=query,
                documents=docs,
                top_n=top_k,
                model="rerank-english-v2.0"
            )
            
            # Map back to original documents
            reranked = []
            for result in rerank_results.results:
                original_doc = documents[result.index]
                reranked.append({
                    "document": original_doc["document"],
                    "score": result.relevance_score,
                    "method": "rerank"
                })
            
            return reranked
            
        except Exception:
            # Fallback to original order if re-ranking fails
            return documents[:top_k]
    
    def retrieve(self, query: str, top_k: int = 5, modality: str | None = None) -> list[dict]:
        """
        Retrieve documents using hybrid search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            modality: Optional modality filter
            
        Returns:
            List of retrieved documents
        """
        cfg = get_model_config()
        use_hybrid = cfg.get("retrieval", {}).get("use_hybrid", False)
        
        if not use_hybrid or not BM25_AVAILABLE:
            # Fall back to vector-only search
            return self.vector_retriever.retrieve(query, top_k=top_k, modality=modality)
        
        # Perform hybrid search
        retrieval_top_k = top_k * 3  # Retrieve more for combination
        
        bm25_results = self._bm25_search(query, top_k=retrieval_top_k)
        vector_results = self._vector_search(query, top_k=retrieval_top_k)
        
        # Combine scores
        combined = self._combine_scores(bm25_results, vector_results)
        
        # Re-rank with Cohere
        reranked = self._rerank_with_cohere(query, combined, top_k=top_k)
        
        # Apply modality filter if specified
        if modality:
            reranked = [
                doc for doc in reranked
                if doc["document"]["metadata"].get("modality") == modality
            ]
        
        # Format output to match standard retriever interface
        formatted_results = []
        for doc in reranked[:top_k]:
            formatted_results.append({
                "id": doc["document"]["id"],
                "content": doc["document"]["content"],
                "metadata": doc["document"]["metadata"],
                "score": doc["score"]
            })
        
        return formatted_results
    
    def rebuild_index(self) -> None:
        """Rebuild BM25 index (call after adding new documents)."""
        self._build_bm25_index()
