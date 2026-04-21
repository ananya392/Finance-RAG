# embeddings/build_index.py

import json
from tqdm import tqdm
from typing import List, Dict

from .encoder import EmbeddingEncoder
from .faiss_utils import FAISSIndex


def load_documents(file_path: str) -> List[Dict]:
    """
    Expected format:
    [
        {"id": 1, "text": "...", "source": "..."},
        ...
    ]
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_documents(documents: List[Dict], chunk_size: int = 200) -> List[Dict]:
    """
    Simple text chunking
    """
    chunks = []

    for doc in documents:
        text = doc["text"]
        words = text.split()

        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i:i + chunk_size])

            chunks.append({
                "text": chunk_text,
                "source": doc.get("source"),
                "doc_id": doc.get("id")
            })

    return chunks


def build_faiss_index(data_path: str):
    print("Loading documents...")
    documents = load_documents(data_path)

    print("Chunking documents...")
    chunks = chunk_documents(documents)

    texts = [chunk["text"] for chunk in chunks]

    print("Encoding embeddings...")
    encoder = EmbeddingEncoder()
    embeddings = encoder.encode(texts)

    dim = embeddings.shape[1]
    index = FAISSIndex(dim)

    print("Training index...")
    index.train(embeddings)

    print("Adding embeddings...")
    index.add(embeddings, chunks)

    print("Saving index...")
    index.save()

    print("✅ FAISS index built successfully!")


if __name__ == "__main__":
    build_faiss_index("data/processed_chunks/news.json")