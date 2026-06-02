from __future__ import annotations

from typing import Any

from src.utils.config import get_model_config, get_settings

try:
    from litellm import completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


class LLMHandler:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.cfg = get_model_config().get("llm", {})

    def generate(self, query: str, contexts: list[dict[str, Any]]) -> str:
        provider = self.settings.llm_provider or self.cfg.get("provider", "local")
        
        # Check if we have image contexts that require vision-language model
        has_images = any(
            ctx.get("metadata", {}).get("modality") == "image" or
            ctx.get("metadata", {}).get("is_image", False)
            for ctx in contexts
        )
        
        # Use vision-language model if available and we have images
        if has_images and LITELLM_AVAILABLE and self.settings.openai_api_key:
            return self._generate_vision(query, contexts)
        
        if provider == "openai" and self.settings.openai_api_key:
            return self._generate_openai(query, contexts)
        return self._generate_local(query, contexts)

    def _generate_local(self, query: str, contexts: list[dict[str, Any]]) -> str:
        if not contexts:
            return (
                "No relevant documents were found in the knowledge base. "
                "Try ingesting PDFs, images, or audio files first."
            )

        blocks = []
        for i, ctx in enumerate(contexts, start=1):
            meta = ctx.get("metadata", {})
            source = meta.get("filename") or meta.get("source", "unknown")
            modality = meta.get("modality", "unknown")
            blocks.append(
                f"[{i}] ({modality}) {source}\n{ctx.get('content', '')}"
            )

        context_block = "\n\n".join(blocks)
        return (
            f"Question: {query}\n\n"
            f"Based on the retrieved context:\n\n{context_block}\n\n"
            "Answer (synthesized from sources above): "
            + self._summarize_from_context(query, contexts)
        )

    def _summarize_from_context(
        self,
        query: str,
        contexts: list[dict[str, Any]],
    ) -> str:
        top = contexts[0].get("content", "")[:800]
        return (
            f"The most relevant passage suggests: {top[:400]}... "
            f"(Configure OPENAI_API_KEY and LLM_PROVIDER=openai for full generative answers.)"
        )

    def _generate_openai(self, query: str, contexts: list[dict[str, Any]]) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.settings.openai_api_key)
        context_text = "\n\n".join(
            f"Source: {c.get('metadata', {}).get('filename', 'doc')}\n{c.get('content', '')}"
            for c in contexts
        )
        model = self.cfg.get("openai_model", "gpt-4o-mini")
        response = client.chat.completions.create(
            model=model,
            temperature=self.cfg.get("temperature", 0.2),
            max_tokens=self.cfg.get("max_tokens", 1024),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer using only the provided context. "
                        "If the context is insufficient, say so clearly."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_text}\n\nQuestion: {query}",
                },
            ],
        )
        return response.choices[0].message.content or ""

    def _generate_vision(self, query: str, contexts: list[dict[str, Any]]) -> str:
        """Generate response using vision-language model for multimodal synthesis."""
        if not contexts:
            return "No relevant documents were found in the knowledge base."
        
        try:
            # Separate text and image contexts
            text_contexts = []
            image_contexts = []
            
            for ctx in contexts:
                meta = ctx.get("metadata", {})
                if meta.get("modality") == "image" or meta.get("is_image", False):
                    image_contexts.append(ctx)
                else:
                    text_contexts.append(ctx)
            
            # Build text context block
            text_blocks = []
            for i, ctx in enumerate(text_contexts, start=1):
                meta = ctx.get("metadata", {})
                source = meta.get("filename") or meta.get("source", "unknown")
                text_blocks.append(
                    f"[{i}] {source}\n{ctx.get('content', '')}"
                )
            
            # Build image context block
            image_blocks = []
            for i, ctx in enumerate(image_contexts, start=1):
                meta = ctx.get("metadata", {})
                source = meta.get("filename") or meta.get("source", "unknown")
                caption = ctx.get("content", "")
                image_blocks.append(f"[{i}] {source}: {caption}")
            
            # Prepare messages for vision model
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful corporate assistant. Answer the user's question "
                        "based strictly on the attached text context and image context. "
                        "Provide citations for your sources."
                    ),
                }
            ]
            
            # Add text context
            if text_blocks:
                messages.append({
                    "role": "user",
                    "content": f"Text Context:\n\n" + "\n\n".join(text_blocks)
                })
            
            # Add image context (for now, use captions since we don't have actual image URLs)
            if image_blocks:
                messages.append({
                    "role": "user",
                    "content": f"Image Context:\n\n" + "\n\n".join(image_blocks)
                })
            
            # Add the query
            messages.append({
                "role": "user",
                "content": f"Question: {query}"
            })
            
            # Use litellm to call vision model
            vision_model = self.cfg.get("vision_model", "gpt-4o")
            response = completion(
                model=vision_model,
                messages=messages,
                temperature=self.cfg.get("temperature", 0.2),
                max_tokens=self.cfg.get("max_tokens", 1024),
                api_key=self.settings.openai_api_key,
            )
            
            return response.choices[0].message.content or ""
            
        except Exception as e:
            # Fallback to text-only generation if vision fails
            return self._generate_openai(query, contexts)
