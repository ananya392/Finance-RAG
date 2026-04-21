# reasoning/answer_generator.py

from .gemini_client import GeminiClient
from .prompt_templates import build_analysis_prompt


class AnswerGenerator:
    def __init__(self, api_key: str):
        self.client = GeminiClient(api_key)

    def generate(self, query: str, context: dict) -> str:
        prompt = build_analysis_prompt(query, context)
        answer = self.client.generate(prompt)
        return answer