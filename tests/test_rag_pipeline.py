import pytest
from langchain_core.documents import Document

from src.rag_pipeline import RAGPipeline


@pytest.fixture
def rag(monkeypatch):
    """
    EN: Build a RAGPipeline but mock external dependencies (LLM + retrieval).
    FR: Tests stables => pas d'appel API, pas de dépendance réseau.
    """
    pipeline = RAGPipeline()

    # 1) Mock retrieval: return deterministic docs
    def fake_search(query: str, k: int = 10):
        docs = [
            Document(page_content="Revenue 2023: 86,153 million euros.", metadata={"page": 10}),
            Document(page_content="2023 growth was 13% compared to 2022.", metadata={"page": 11}),
            Document(page_content="LVMH has more than 5,000 stores worldwide.", metadata={"page": 49}),
        ]
        # Return (doc, score)
        return [(docs[0], 0.90), (docs[1], 0.85), (docs[2], 0.80)]

    monkeypatch.setattr(pipeline.vector_store, "search", fake_search)

    # 2) Mock reranker: keep the same order for test simplicity
    def fake_rerank(query, docs_scores, top_k=5):
        return docs_scores[:top_k]

    monkeypatch.setattr(pipeline.reranker, "rerank", fake_rerank)

    # 3) Mock LLM generation: return a fixed answer (no API call)
    def fake_generate(context: str, question: str) -> str:
        return "LVMH revenue in 2023 was 86,153 million euros [Page 10]."

    monkeypatch.setattr(pipeline.llm, "generate", fake_generate)

    return pipeline


def test_rag_init(rag):
    """
    EN: Sanity check that components exist.
    FR: Vérifie l'init.
    """
    assert rag.vector_store is not None
    assert rag.reranker is not None
    assert rag.llm is not None


def test_query_returns_expected_shape(rag):
    """
    EN: Query returns a structured result.
    FR: Structure standard: answer + sources + latency + from_cache.
    """
    result = rag.query("What was LVMH revenue in 2023?", use_cache=False)

    assert "answer" in result
    assert "sources" in result
    assert "latency_ms" in result
    assert "from_cache" in result

    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0
    assert isinstance(result["sources"], list)


def test_cache_behavior(rag):
    """
    EN: Cache should return the same answer on the second call.
    FR: Test important pour la démo et la latence.
    """
    q = "What was LVMH revenue in 2023?"

    r1 = rag.query(q, use_cache=True)
    assert r1["from_cache"] is False

    r2 = rag.query(q, use_cache=True)
    assert r2["from_cache"] is True
    assert r2["answer"] == r1["answer"]


def test_metrics_update(rag):
    """
    EN: Metrics should be updated after a query.
    FR: Vérifie que total_queries augmente et que avg_latency_ms existe.
    """
    rag.query("Test metrics", use_cache=False)
    metrics = rag.get_metrics()

    assert metrics["total_queries"] >= 1
    assert "avg_latency_ms" in metrics
    assert "cache_hit_rate" in metrics
