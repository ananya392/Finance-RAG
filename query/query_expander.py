# query/query_expander.py

from typing import List


FINANCIAL_SYNONYMS = {
    "interest rates": ["fed rates", "bond yields", "monetary policy"],
    "inflation": ["cpi", "price rise", "cost of living"],
    "stock market": ["equities", "indices", "shares"],
    "tech stocks": ["nasdaq", "technology sector", "growth stocks"]
}


class QueryExpander:
    def expand(self, query: str) -> List[str]:
        expanded_queries = [query]

        for key, values in FINANCIAL_SYNONYMS.items():
            if key in query.lower():
                expanded_queries.extend(values)

        return list(set(expanded_queries))