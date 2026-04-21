# reasoning/context_builder.py

from typing import List, Dict


class ContextBuilder:
    def __init__(self):
        pass

    def build(self, retrieved_docs: List[Dict], market_data: Dict) -> Dict:
        """
        Combine retrieved documents + structured market signals
        """

        # Limit top documents
        top_docs = retrieved_docs[:5]

        news_context = [doc["text"] for doc in top_docs]

        return {
            "news": news_context,
            "market_data": market_data
        }