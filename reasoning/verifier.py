# reasoning/verifier.py

from .gemini_client import GeminiClient


class AnswerVerifier:
    def __init__(self, api_key: str):
        self.client = GeminiClient(api_key)

    def verify(self, query: str, answer: str, context: dict) -> str:
        news_text = "\n".join(context["news"])

        prompt = f"""
You are a strict financial auditor.

Query:
{query}

Answer:
{answer}

Context:
{news_text}

Task:
- Identify any claims in the answer NOT supported by the context
- If unsupported claims exist, correct the answer
- If fully grounded, return the original answer

Return ONLY the corrected or verified answer.
"""

        verified = self.client.generate(prompt)
        return verified