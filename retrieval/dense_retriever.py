# retrieval/dense_retriever.py

from embeddings.encoder import EmbeddingEncoder
from embeddings.faiss_utils import FAISSIndex


class DenseRetriever:
    def __init__(self):
        self.encoder = EmbeddingEncoder()
        self.index = FAISSIndex(dim=384)  # MiniLM dim
        self.index.load()

    def retrieve(self, query: str, k: int = 20):
        """
        Returns top-k semantic results
        """
        query_vec = self.encoder.encode_query(query)
        scores, results = self.index.search(query_vec, k)

        return [
            {
                "text": r["text"],
                "score": float(scores[i]),
                "source": r.get("source"),
                "doc_id": r.get("doc_id")
            }
            for i, r in enumerate(results)
        ]