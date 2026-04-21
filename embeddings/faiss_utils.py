# embeddings/faiss_utils.py

import faiss
import numpy as np
import os
import pickle
from typing import Tuple
from .config import INDEX_TYPE, N_LIST, N_PROBE, FAISS_INDEX_PATH, METADATA_PATH


class FAISSIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = self._create_index()
        self.metadata = []

    def _create_index(self):
        if INDEX_TYPE == "FLAT":
            index = faiss.IndexFlatIP(self.dim)  # cosine similarity (normalized)
        elif INDEX_TYPE == "IVF":
            quantizer = faiss.IndexFlatIP(self.dim)
            index = faiss.IndexIVFFlat(quantizer, self.dim, N_LIST, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError("Unsupported index type")

        return index

    def train(self, embeddings: np.ndarray):
        """
        Required for IVF index
        """
        if isinstance(self.index, faiss.IndexIVF):
            self.index.train(embeddings)

    def add(self, embeddings: np.ndarray, metadata: list):
        """
        Add embeddings + metadata
        """
        if isinstance(self.index, faiss.IndexIVF):
            self.index.nprobe = N_PROBE

        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[np.ndarray, list]:
        """
        Search top-k similar documents
        """
        query_vector = np.expand_dims(query_vector, axis=0)

        scores, indices = self.index.search(query_vector, k)

        results = []
        for idx in indices[0]:
            if idx < len(self.metadata):
                results.append(self.metadata[idx])

        return scores[0], results

    def save(self):
        """
        Save index + metadata
        """
        os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

        faiss.write_index(self.index, FAISS_INDEX_PATH)

        with open(METADATA_PATH, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        """
        Load index + metadata
        """
        self.index = faiss.read_index(FAISS_INDEX_PATH)

        with open(METADATA_PATH, "rb") as f:
            self.metadata = pickle.load(f)