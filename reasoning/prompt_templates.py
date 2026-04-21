# reasoning/prompt_templates.py

def build_analysis_prompt(query: str, context: dict) -> str:
    news_text = "\n".join(context["news"])

    market_info = "\n".join(
        [f"{k}: {v}" for k, v in context["market_data"].items()]
    )

    return f"""
You are a financial analyst.

User Query:
{query}

Context:
---------------------
NEWS:
{news_text}

MARKET DATA:
{market_info}
---------------------

Instructions:
1. Explain what happened in the market
2. Identify key drivers using ONLY the provided context
3. Connect news events with market movements
4. Provide short-term vs long-term implications
5. Do NOT hallucinate or add external knowledge

Answer clearly and analytically.
"""