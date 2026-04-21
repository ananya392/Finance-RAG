# query/rewrite.py

import google.generativeai as genai
from typing import Dict


class QueryRewriter:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-pro")

    def rewrite(self, query: str) -> str:
        prompt = f"""
You are a financial query optimizer.

Rewrite the following query to make it:
- More specific
- Financially precise
- Suitable for document retrieval

Query: "{query}"

Return ONLY the rewritten query.
"""

        response = self.model.generate_content(prompt)
        return response.text.strip()