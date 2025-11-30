"""
Embedding service (stub) using Google embeddings.
Mask PII before embedding and support batch operations.
"""
from typing import List
import os
import google.generativeai as genai
from services.pii_masking import PIIMaskingService


class EmbeddingService:
    def __init__(self, pii_masker: PIIMaskingService | None = None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        genai.configure(api_key=api_key)
        self.model_name = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
        self.pii_masker = pii_masker or PIIMaskingService()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        masked = [self.pii_masker.mask_text(t or "") for t in texts]
        # Batch call embeddings API
        res = genai.embed_content(
            model=self.model_name,
            content=masked,
        )
        # genai.embed_content returns dict; normalize vectors
        if isinstance(res, dict) and "embedding" in res:
            # Single
            return [res["embedding"]["values"]]
        if isinstance(res, dict) and "embeddings" in res:
            return [e["values"] for e in res["embeddings"]]
        # Fallback
        return []


