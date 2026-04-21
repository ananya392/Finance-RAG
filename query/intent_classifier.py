# query/intent_classifier.py

from typing import Dict


class QueryIntentClassifier:
    def classify(self, query: str) -> Dict:
        query_lower = query.lower()

        if any(word in query_lower for word in ["why", "reason", "cause", "impact"]):
            intent = "explanatory"

        elif any(word in query_lower for word in ["price", "stock", "market", "index"]):
            intent = "market"

        elif any(word in query_lower for word in ["eps", "revenue", "profit", "ratio"]):
            intent = "fundamental"

        else:
            intent = "general"

        return {
            "intent": intent,
            "confidence": 0.7  # simple heuristic
        }