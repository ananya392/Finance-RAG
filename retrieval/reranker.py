# retrieval/reranker.py

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query: str, documents: list, top_k: int = 10):
        """
        Re-rank documents using cross-encoder
        """
        pairs = [(query, doc["text"]) for doc in documents]

        scores = self.model.predict(pairs)

        for i, doc in enumerate(documents):
            doc["rerank_score"] = float(scores[i])

        ranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)

        return ranked[:top_k]