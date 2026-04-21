# retrieval/sparse_retriever.py

from rank_bm25 import BM25Okapi
import json
import numpy as np


class SparseRetriever:
    def __init__(self, corpus_path: str):
        self.documents = self._load_corpus(corpus_path)
        self.texts = [doc["text"] for doc in self.documents]
        self.tokenized_corpus = [text.split() for text in self.texts]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _load_corpus(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def retrieve(self, query: str, k: int = 20):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)

        top_k_idx = np.argsort(scores)[::-1][:k]

        return [
            {
                "text": self.documents[i]["text"],
                "score": float(scores[i]),
                "source": self.documents[i].get("source"),
                "doc_id": self.documents[i].get("doc_id")
            }
            for i in top_k_idx
        ]