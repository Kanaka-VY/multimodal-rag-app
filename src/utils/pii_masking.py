"""
PII Anonymization and De-anonymization for Enterprise Data Privacy
Uses Microsoft Presidio to mask PII before sending to external APIs.
"""

import json
import re
from typing import Any

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import RecognizerResult, OperatorConfig
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False

from src.utils.config import get_model_config


class PIIMasker:
    """PII masking and de-masking for enterprise data privacy."""
    
    def __init__(self):
        self.analyzer = None
        self.anonymizer = None
        self.pii_mapping = {}  # Store mapping for de-anonymization
        
        if PRESIDIO_AVAILABLE:
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
        
        cfg = get_model_config()
        self.enabled = cfg.get("privacy", {}).get("enable_pii_masking", True)
        self.mask_entities = cfg.get("privacy", {}).get(
            "mask_entities",
            ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "IBAN_CODE", "CREDIT_CARD"]
        )
        self.mask_strategy = cfg.get("privacy", {}).get("mask_strategy", "replace")
    
    def analyze_pii(self, text: str) -> list[RecognizerResult]:
        """
        Analyze text for PII entities.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of RecognizerResult with PII entities
        """
        if not self.enabled or not self.analyzer:
            return []
        
        try:
            results = self.analyzer.analyze(
                text=text,
                entities=self.mask_entities,
                language="en"
            )
            return results
        except Exception:
            return []
    
    def mask_text(self, text: str, user_id: str | None = None) -> tuple[str, dict]:
        """
        Mask PII in text.
        
        Args:
            text: Text to mask
            user_id: User ID for storing mapping (for de-anonymization)
            
        Returns:
            Tuple of (masked_text, pii_mapping)
        """
        if not self.enabled or not self.analyzer or not self.anonymizer:
            return text, {}
        
        # Analyze PII
        pii_results = self.analyze_pii(text)
        
        if not pii_results:
            return text, {}
        
        # Create anonymization config
        operators = {}
        for entity in self.mask_entities:
            if self.mask_strategy == "replace":
                operators[entity] = OperatorConfig("replace", {"new_value": f"[{entity}]"})
            elif self.mask_strategy == "hash":
                operators[entity] = OperatorConfig("hash", {})
            elif self.mask_strategy == "mask":
                operators[entity] = OperatorConfig("mask", {"chars_to_mask": 4, "masking_char": "*"})
            else:
                operators[entity] = OperatorConfig("replace", {"new_value": f"[{entity}]"})
        
        # Anonymize
        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=pii_results,
            operators=operators
        )
        
        masked_text = anonymized_result.text
        
        # Store mapping for de-anonymization
        pii_mapping = {}
        for result in pii_results:
            original_value = text[result.start:result.end]
            masked_value = masked_text[result.start:result.end] if result.start < len(masked_text) else f"[{result.entity_type}]"
            
            pii_mapping[str(result.start)] = {
                "original": original_value,
                "masked": masked_value,
                "entity_type": result.entity_type,
                "confidence": result.score
            }
        
        # Store mapping by user if provided
        if user_id:
            if user_id not in self.pii_mapping:
                self.pii_mapping[user_id] = {}
            self.pii_mapping[user_id].update(pii_mapping)
        
        return masked_text, pii_mapping
    
    def unmask_text(self, masked_text: str, user_id: str) -> str:
        """
        De-anonymize text for authenticated user.
        
        Args:
            masked_text: Text with masked PII
            user_id: User ID for retrieving mapping
            
        Returns:
            De-anonymized text
        """
        if not self.enabled or user_id not in self.pii_mapping:
            return masked_text
        
        # Retrieve mapping
        user_mapping = self.pii_mapping.get(user_id, {})
        
        # Replace masked values with original values
        unmasked_text = masked_text
        for position, mapping in user_mapping.items():
            original = mapping["original"]
            masked = mapping["masked"]
            unmasked_text = unmasked_text.replace(masked, original)
        
        return unmasked_text
    
    def mask_document_chunks(self, chunks: list[str], user_id: str | None = None) -> tuple[list[str], dict]:
        """
        Mask PII in document chunks.
        
        Args:
            chunks: List of text chunks
            user_id: User ID for storing mapping
            
        Returns:
            Tuple of (masked_chunks, combined_pii_mapping)
        """
        masked_chunks = []
        combined_mapping = {}
        
        for chunk in chunks:
            masked_chunk, mapping = self.mask_text(chunk, user_id)
            masked_chunks.append(masked_chunk)
            combined_mapping.update(mapping)
        
        return masked_chunks, combined_mapping
    
    def clear_user_mapping(self, user_id: str) -> None:
        """
        Clear PII mapping for a user (for cleanup).
        
        Args:
            user_id: User ID to clear
        """
        if user_id in self.pii_mapping:
            del self.pii_mapping[user_id]


class PrivacyMiddleware:
    """Middleware for integrating PII masking into the pipeline."""
    
    def __init__(self):
        self.masker = PIIMasker()
    
    def preprocess_query(self, query: str, user_id: str | None = None) -> tuple[str, dict]:
        """
        Preprocess user query by masking PII.
        
        Args:
            query: User query
            user_id: User ID for mapping
            
        Returns:
            Tuple of (masked_query, pii_mapping)
        """
        return self.masker.mask_text(query, user_id)
    
    def preprocess_contexts(
        self,
        contexts: list[dict],
        user_id: str | None = None
    ) -> tuple[list[dict], dict]:
        """
        Preprocess retrieved contexts by masking PII.
        
        Args:
            contexts: List of context dictionaries
            user_id: User ID for mapping
            
        Returns:
            Tuple of (masked_contexts, pii_mapping)
        """
        masked_contexts = []
        combined_mapping = {}
        
        for ctx in contexts:
            content = ctx.get("content", "")
            masked_content, mapping = self.masker.mask_text(content, user_id)
            
            masked_ctx = ctx.copy()
            masked_ctx["content"] = masked_content
            masked_ctx["original_content"] = content  # Store original for reference
            
            masked_contexts.append(masked_ctx)
            combined_mapping.update(mapping)
        
        return masked_contexts, combined_mapping
    
    def postprocess_answer(
        self,
        answer: str,
        user_id: str
    ) -> str:
        """
        Postprocess generated answer by de-anonymizing for authenticated user.
        
        Args:
            answer: Generated answer
            user_id: User ID for retrieving mapping
            
        Returns:
            De-anonymized answer
        """
        return self.masker.unmask_text(answer, user_id)


# Singleton instance
_privacy_middleware = None

def get_privacy_middleware() -> PrivacyMiddleware:
    """Get the singleton privacy middleware instance."""
    global _privacy_middleware
    if _privacy_middleware is None:
        _privacy_middleware = PrivacyMiddleware()
    return _privacy_middleware
