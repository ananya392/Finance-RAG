# pipelines/rag_pipeline.py

from typing import Dict, List

# Query
from query import QueryRewriter, QueryIntentClassifier, QueryExpander

# Retrieval
from retrieval import (
    DenseRetriever,
    SparseRetriever,
    reciprocal_rank_fusion,
    CrossEncoderReranker
)

# Reasoning
from reasoning import (
    ContextBuilder,
    AnswerGenerator,
    AnswerVerifier
)


class FinancialRAGPipeline:
    def __init__(self, gemini_api_key: str, corpus_path: str):
        # Query modules
        self.rewriter = QueryRewriter(gemini_api_key)
        self.classifier = QueryIntentClassifier()
        self.expander = QueryExpander()

        # Retrieval modules
        self.dense = DenseRetriever()
        self.sparse = SparseRetriever(corpus_path)
        self.reranker = CrossEncoderReranker()

        # Reasoning modules
        self.context_builder = ContextBuilder()
        self.generator = AnswerGenerator(gemini_api_key)
        self.verifier = AnswerVerifier(gemini_api_key)

    # -------------------------------
    # 🔷 Step 1: Query Understanding
    # -------------------------------
    def process_query(self, query: str) -> Dict:
        rewritten = self.rewriter.rewrite(query)
        intent = self.classifier.classify(rewritten)
        expanded = self.expander.expand(rewritten)

        return {
            "original": query,
            "rewritten": rewritten,
            "intent": intent,
            "expanded": expanded
        }

    # -------------------------------
    # 🔷 Step 2: Retrieval
    # -------------------------------
    def retrieve(self, query_bundle: Dict) -> List[Dict]:
        rewritten = query_bundle["rewritten"]
        expanded_queries = query_bundle["expanded"]

        # Dense retrieval → rewritten query
        dense_results = self.dense.retrieve(rewritten, k=20)

        # Sparse retrieval → expanded queries
        sparse_results = []
        for q in expanded_queries:
            sparse_results.extend(self.sparse.retrieve(q, k=10))

        # Hybrid fusion
        fused = reciprocal_rank_fusion(dense_results, sparse_results)

        # Re-ranking
        reranked = self.reranker.rerank(rewritten, fused, top_k=5)

        return reranked

    # -------------------------------
    # 🔷 Step 3: Context Building
    # -------------------------------
    def build_context(self, retrieved_docs: List[Dict], market_data: Dict) -> Dict:
        return self.context_builder.build(retrieved_docs, market_data)

    # -------------------------------
    # 🔷 Step 4: Reasoning
    # -------------------------------
    def generate_answer(self, query: str, context: Dict) -> str:
        return self.generator.generate(query, context)

    # -------------------------------
    # 🔷 Step 5: Verification
    # -------------------------------
    def verify_answer(self, query: str, answer: str, context: Dict) -> str:
        return self.verifier.verify(query, answer, context)

    # -------------------------------
    # 🔥 FULL PIPELINE
    # -------------------------------
    def run(self, query: str, market_data: Dict) -> Dict:
        print("🔹 Processing query...")
        query_bundle = self.process_query(query)

        print("🔹 Retrieving documents...")
        retrieved_docs = self.retrieve(query_bundle)

        print("🔹 Building context...")
        context = self.build_context(retrieved_docs, market_data)

        print("🔹 Generating answer...")
        answer = self.generate_answer(query_bundle["rewritten"], context)

        print("🔹 Verifying answer...")
        final_answer = self.verify_answer(query_bundle["rewritten"], answer, context)
        print("\n--- DEBUG INFO ---")
        print("Rewritten Query:", query_bundle["rewritten"])
        print("Intent:", query_bundle["intent"])
        print("Top Docs:", [doc["text"][:100] for doc in retrieved_docs])
        return {
            "query": query_bundle,
            "retrieved_docs": retrieved_docs,
            "context": context,
            "answer": final_answer
        }