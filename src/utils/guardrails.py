"""
Active Runtime Guardrails for RAG System
Implements real-time faithfulness checking and fallback mechanisms.
"""

from enum import Enum
from typing import Any

from src.utils.config import get_model_config
from src.utils.monitoring import evaluate_faithfulness


class GuardrailAction(Enum):
    """Actions to take when guardrails are triggered."""
    ALLOW = "allow"
    BLOCK = "block"
    RETRY = "retry"
    FALLBACK = "fallback"


class GuardrailResult:
    """Result of guardrail evaluation."""
    def __init__(
        self,
        action: GuardrailAction,
        faithfulness_score: float | None,
        reason: str,
        metadata: dict[str, Any] | None = None
    ):
        self.action = action
        self.faithfulness_score = faithfulness_score
        self.reason = reason
        self.metadata = metadata or {}


class ActiveGuardrails:
    """Active runtime guardrails for RAG responses."""
    
    def __init__(self):
        cfg = get_model_config()
        self.faithfulness_threshold = cfg.get("observability", {}).get(
            "faithfulness_threshold",
            0.80
        )
        self.max_retries = cfg.get("observability", {}).get("max_retries", 2)
        self.enable_fallback = cfg.get("observability", {}).get("enable_fallback", True)
        self.fallback_message = cfg.get("observability", {}).get(
            "fallback_message",
            "I'm sorry, I cannot verify the answer based on the source documents. Please rephrase your question or try a different query."
        )
    
    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[dict],
        retry_count: int = 0
    ) -> GuardrailResult:
        """
        Evaluate response against guardrails.
        
        Args:
            query: User query
            answer: Generated answer
            contexts: Retrieved contexts
            retry_count: Current retry attempt
            
        Returns:
            GuardrailResult with action and metadata
        """
        # Evaluate faithfulness
        faithfulness = evaluate_faithfulness(query, answer, contexts)
        
        if faithfulness is None:
            # If evaluation failed, be conservative and block
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                faithfulness_score=None,
                reason="Faithfulness evaluation failed",
                metadata={"error": "evaluation_failed"}
            )
        
        # Check if faithfulness is above threshold
        if faithfulness >= self.faithfulness_threshold:
            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                faithfulness_score=faithfulness,
                reason=f"Faithfulness score {faithfulness:.3f} meets threshold {self.faithfulness_threshold:.3f}",
                metadata={"score": faithfulness}
            )
        
        # Faithfulness is below threshold
        if retry_count < self.max_retries:
            return GuardrailResult(
                action=GuardrailAction.RETRY,
                faithfulness_score=faithfulness,
                reason=f"Faithfulness score {faithfulness:.3f} below threshold {self.faithfulness_threshold:.3f}. Retry {retry_count + 1}/{self.max_retries}",
                metadata={
                    "score": faithfulness,
                    "retry_count": retry_count + 1,
                    "max_retries": self.max_retries
                }
            )
        
        # Max retries reached, use fallback
        if self.enable_fallback:
            return GuardrailResult(
                action=GuardrailAction.FALLBACK,
                faithfulness_score=faithfulness,
                reason=f"Faithfulness score {faithfulness:.3f} below threshold after {self.max_retries} retries. Using fallback.",
                metadata={
                    "score": faithfulness,
                    "retry_count": retry_count,
                    "fallback_message": self.fallback_message
                }
            )
        
        # Block the response
        return GuardrailResult(
            action=GuardrailAction.BLOCK,
            faithfulness_score=faithfulness,
            reason=f"Faithfulness score {faithfulness:.3f} below threshold {self.faithfulness_threshold:.3f} after {self.max_retries} retries. Response blocked.",
            metadata={
                "score": faithfulness,
                "retry_count": retry_count
            }
        )
    
    def get_fallback_config(self) -> dict[str, Any]:
        """
        Get configuration for retry attempt.
        
        Returns:
            Configuration adjustments for retry
        """
        return {
            "temperature": 0.8,  # Higher temperature for more diverse responses
            "top_k": 10,  # Retrieve more contexts
            "use_rerank": True,  # Enable re-ranking
            "prompt_template": "You are a helpful assistant. Please answer based strictly on the provided context. If you cannot answer based on the context, say so clearly."
        }


class GuardrailMiddleware:
    """Middleware for integrating guardrails into the query pipeline."""
    
    def __init__(self):
        self.guardrails = ActiveGuardrails()
    
    async def process_query(
        self,
        query: str,
        retriever,
        llm_handler,
        max_retries: int = 2
    ) -> dict[str, Any]:
        """
        Process query with active guardrails.
        
        Args:
            query: User query
            retriever: MultimodalRetriever instance
            llm_handler: LLMHandler instance
            max_retries: Maximum number of retry attempts
            
        Returns:
            Response dictionary with answer and guardrail metadata
        """
        retry_count = 0
        last_result = None
        
        while retry_count <= max_retries:
            # Retrieve contexts
            contexts = retriever.retrieve(query, top_k=5, modality=None)
            
            if not contexts:
                return {
                    "answer": "No relevant documents found for your query.",
                    "guardrail": {
                        "action": "block",
                        "reason": "No contexts retrieved"
                    }
                }
            
            # Generate answer
            if retry_count > 0:
                # Use fallback config for retries
                fallback_config = self.guardrails.get_fallback_config()
                answer = llm_handler.generate(
                    query,
                    contexts,
                    temperature=fallback_config.get("temperature", 0.8)
                )
            else:
                answer = llm_handler.generate(query, contexts)
            
            # Evaluate guardrails
            result = self.guardrails.evaluate(
                query,
                answer,
                contexts,
                retry_count=retry_count
            )
            
            last_result = result
            
            if result.action == GuardrailAction.ALLOW:
                return {
                    "answer": answer,
                    "contexts": contexts,
                    "guardrail": {
                        "action": result.action.value,
                        "faithfulness_score": result.faithfulness_score,
                        "reason": result.reason
                    }
                }
            
            elif result.action == GuardrailAction.RETRY:
                retry_count += 1
                continue
            
            elif result.action == GuardrailAction.FALLBACK:
                return {
                    "answer": self.guardrails.fallback_message,
                    "contexts": contexts,
                    "guardrail": {
                        "action": result.action.value,
                        "faithfulness_score": result.faithfulness_score,
                        "reason": result.reason,
                        "original_answer": answer
                    }
                }
            
            elif result.action == GuardrailAction.BLOCK:
                return {
                    "answer": "I'm sorry, but I cannot provide a reliable answer based on the available documents.",
                    "contexts": contexts,
                    "guardrail": {
                        "action": result.action.value,
                        "faithfulness_score": result.faithfulness_score,
                        "reason": result.reason
                    }
                }
        
        # Should not reach here, but fallback
        return {
            "answer": self.guardrails.fallback_message,
            "guardrail": {
                "action": "fallback",
                "reason": "Max retries exceeded"
            }
        }


# Singleton instance
_guardrail_middleware = None

def get_guardrail_middleware() -> GuardrailMiddleware:
    """Get the singleton guardrail middleware instance."""
    global _guardrail_middleware
    if _guardrail_middleware is None:
        _guardrail_middleware = GuardrailMiddleware()
    return _guardrail_middleware
