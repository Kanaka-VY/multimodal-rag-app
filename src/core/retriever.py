from __future__ import annotations

from typing import Any

from src.core.embeddings import embed_image_query, embed_text_query
from src.database.vector_store import VectorStore
from src.utils.config import get_model_config, get_settings

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False


class MultimodalRetriever:
    def __init__(self, store: VectorStore | None = None) -> None:
        self.store = store or VectorStore()
        self._cfg = get_model_config().get("retrieval", {})
        self._bm25_index = None
        self._bm25_corpus = []
        self._bm25_metadata = []
        self._cohere_client = None
        self._initialize_bm25()
        self._initialize_cohere()

    def _initialize_bm25(self) -> None:
        """Initialize BM25 index with existing documents if hybrid search is enabled."""
        if not BM25_AVAILABLE or not self._cfg.get("use_hybrid", False):
            return
        
        try:
            # Fetch all documents from the store
            all_docs = self.store._collection.get(include=["documents", "metadatas"])
            if all_docs and all_docs["documents"]:
                self._bm25_corpus = all_docs["documents"]
                self._bm25_metadata = all_docs["metadatas"]
                # Tokenize documents for BM25
                tokenized_corpus = [doc.split() for doc in self._bm25_corpus]
                k1 = self._cfg.get("bm25_k1", 1.5)
                b = self._cfg.get("bm25_b", 0.75)
                self._bm25_index = BM25Okapi(tokenized_corpus, k1=k1, b=b)
        except Exception:
            # BM25 initialization failed, will fall back to vector-only search
            self._bm25_index = None

    def _initialize_cohere(self) -> None:
        """Initialize Cohere client for re-ranking if enabled."""
        if not COHERE_AVAILABLE or not self._cfg.get("use_rerank", False):
            return
        
        try:
            settings = get_settings()
            api_key = getattr(settings, 'cohere_api_key', None)
            if api_key:
                self._cohere_client = cohere.Client(api_key)
        except Exception:
            self._cohere_client = None

    def retrieve(
        self,
        query: str,
        modality: str | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        k = top_k or self._cfg.get("top_k", 5)
        threshold = self._cfg.get("score_threshold", 0.3)

        where = {"modality": modality} if modality else None
        embedding = (
            embed_image_query(query)
            if modality == "image"
            else embed_text_query(query)
        )

        if modality is None:
            # Hybrid search: combine BM25 and vector search
            if self._cfg.get("use_hybrid", False) and self._bm25_index:
                results = self._hybrid_retrieve(query, k, threshold)
            else:
                # Fallback to vector-only search
                text_hits = self._parse_hits(
                    self.store.query(embed_text_query(query), top_k=k),
                    threshold,
                )
                image_hits = self._parse_hits(
                    self.store.query(embed_image_query(query), top_k=k),
                    threshold,
                )
                merged = text_hits + image_hits
                merged.sort(key=lambda x: x["score"], reverse=True)
                results = merged[:k]
            
            # Apply re-ranking if enabled
            if self._cfg.get("use_rerank", False) and self._cohere_client:
                results = self._rerank(query, results, k)
            
            return results

        raw = self.store.query(embedding, top_k=k, where=where)
        results = self._parse_hits(raw, threshold)
        
        # Apply re-ranking if enabled
        if self._cfg.get("use_rerank", False) and self._cohere_client:
            results = self._rerank(query, results, k)
        
        return results

    def _hybrid_retrieve(self, query: str, top_k: int, threshold: float) -> list[dict[str, Any]]:
        """Combine BM25 sparse search with vector dense search."""
        # BM25 sparse search
        tokenized_query = query.split()
        bm25_scores = self._bm25_index.get_scores(tokenized_query)
        
        # Get top BM25 results
        bm25_top_indices = bm25_scores.argsort()[-top_k*2:][::-1]  # Get more for reranking
        bm25_hits = []
        for idx in bm25_top_indices:
            if idx < len(self._bm25_corpus):
                bm25_hits.append({
                    "content": self._bm25_corpus[idx],
                    "metadata": self._bm25_metadata[idx] or {},
                    "bm25_score": float(bm25_scores[idx]),
                })
        
        # Vector dense search
        vector_hits_text = self._parse_hits(
            self.store.query(embed_text_query(query), top_k=top_k*2),
            threshold,
        )
        vector_hits_image = self._parse_hits(
            self.store.query(embed_image_query(query), top_k=top_k*2),
            threshold,
        )
        
        # Combine and normalize scores
        combined = {}
        for hit in bm25_hits:
            doc_id = hit["metadata"].get("source", "") + str(hash(hit["content"]))
            combined[doc_id] = {
                "content": hit["content"],
                "metadata": hit["metadata"],
                "bm25_score": hit["bm25_score"],
                "vector_score": 0.0,
            }
        
        for hit in vector_hits_text + vector_hits_image:
            doc_id = hit["metadata"].get("source", "") + str(hash(hit["content"]))
            if doc_id in combined:
                combined[doc_id]["vector_score"] = hit["score"]
            else:
                combined[doc_id] = {
                    "content": hit["content"],
                    "metadata": hit["metadata"],
                    "bm25_score": 0.0,
                    "vector_score": hit["score"],
                }
        
        # Calculate combined score (weighted average)
        results = []
        for doc_id, data in combined.items():
            # Normalize scores to 0-1 range
            bm25_norm = min(data["bm25_score"] / (max(bm25_scores) if bm25_scores.max() > 0 else 1), 1.0)
            vector_norm = data["vector_score"]
            
            # Weighted combination (70% vector, 30% BM25)
            combined_score = 0.7 * vector_norm + 0.3 * bm25_norm
            
            if combined_score >= threshold:
                results.append({
                    "content": data["content"],
                    "metadata": data["metadata"],
                    "score": round(combined_score, 4),
                })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _rerank(self, query: str, results: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Re-rank results using Cohere's re-ranking API."""
        if not self._cohere_client or not results:
            return results
        
        try:
            # Prepare documents for re-ranking
            documents = [result["content"] for result in results]
            
            # Get more results for re-ranking, then trim to top_k
            rerank_top_k = self._cfg.get("rerank_top_k", top_k * 2)
            
            # Call Cohere re-rank API
            rerank_response = self._cohere_client.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=documents,
                top_n=rerank_top_k,
            )
            
            # Reorder results based on re-ranking scores
            reranked = []
            for result in rerank_response.results:
                original_index = result.index
                if original_index < len(results):
                    reranked_item = results[original_index].copy()
                    reranked_item["score"] = round(result.relevance_score, 4)
                    reranked_item["reranked"] = True
                    reranked.append(reranked_item)
            
            return reranked[:top_k]
        except Exception:
            # Re-ranking failed, return original results
            return results[:top_k]

    def _parse_hits(
        self,
        raw: dict[str, Any],
        threshold: float,
    ) -> list[dict[str, Any]]:
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]

        hits: list[dict[str, Any]] = []
        for doc, meta, dist in zip(docs, metas, dists):
            score = 1.0 - float(dist)
            if score < threshold:
                continue
            hits.append(
                {
                    "content": doc,
                    "metadata": meta or {},
                    "score": round(score, 4),
                }
            )
        hits.sort(key=lambda x: x["score"], reverse=True)
        return hits
