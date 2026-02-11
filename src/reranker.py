from typing import List, Tuple
from langchain_core.documents import Document

class Reranker:
    def __init__(self):
        # Poids équilibrés (empiriques)
        self.sim_weight = 0.6
        self.keyword_weight = 0.2
        self.length_weight = 0.1
        self.financial_weight = 0.1

    def rerank(
        self,
        query: str,
        docs_scores: List[Tuple[Document, float]],
        top_k: int = 5
    ) -> List[Tuple[Document, float]]:

        rescored = []

        query_lower = query.lower()
        query_words = set(query_lower.split())

        for doc, sim_score in docs_scores:
            text = doc.page_content.lower()

            # Keyword overlap
            doc_words = set(text.split())
            keyword_score = (
                len(query_words & doc_words) / len(query_words)
                if query_words else 0.0
            )

            # Length score (ideal ~150–300 words)
            wc = len(doc.page_content.split())
            length_score = 1.0 / (1.0 + abs(wc - 200) / 200)

            # 🔥 FINANCIAL BOOST
            financial_boost = 0.0
            if any(k in text for k in [
                "revenue", "net sales", "sales",
                "eur", "€", "million", "billion"
            ]):
                financial_boost += 0.15

            if doc.metadata.get("has_numbers", False):
                financial_boost += 0.10

            final_score = (
                self.sim_weight * sim_score +
                self.keyword_weight * keyword_score +
                self.length_weight * length_score +
                self.financial_weight * financial_boost
            )

            rescored.append((doc, final_score))

        rescored.sort(key=lambda x: x[1], reverse=True)
        return rescored[:top_k]
