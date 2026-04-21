# embeddings/encoder.py

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from .config import MODEL_NAME, BATCH_SIZE


class EmbeddingEncoder:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        """
        embeddings = self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # important for cosine similarity
        )
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query
        """
        return self.encode([query])[0]