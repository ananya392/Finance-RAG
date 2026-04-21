# retrieval/hybrid_fusion.py

from collections import defaultdict


def reciprocal_rank_fusion(dense_results, sparse_results, k: int = 60):
    """
    RRF Fusion
    """
    scores = defaultdict(float)
    doc_map = {}

    # Dense rankings
    for rank, doc in enumerate(dense_results):
        key = doc["text"]
        scores[key] += 1 / (k + rank)
        doc_map[key] = doc

    # Sparse rankings
    for rank, doc in enumerate(sparse_results):
        key = doc["text"]
        scores[key] += 1 / (k + rank)
        doc_map[key] = doc

    # Sort
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [
        {
            **doc_map[text],
            "fusion_score": score
        }
        for text, score in ranked
    ]