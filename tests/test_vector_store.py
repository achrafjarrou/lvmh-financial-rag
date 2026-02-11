import pytest
from src.vector_store import VectorStore


@pytest.fixture
def vector_store():
    """
    EN: If a Chroma DB already exists locally, load it.
    FR: On ne crée pas automatiquement la DB en test (peut être long).
    """
    store = VectorStore()
    if not store.exists():
        pytest.skip("Chroma DB not found locally. Run VectorStore.create() once to enable this test.")
    return store


def test_vector_store_exists_and_loaded(vector_store):
    """
    EN: DB exists and is loaded.
    FR: Vérifie que la DB est bien chargée.
    """
    assert vector_store.exists()
    assert vector_store.db is not None


def test_search_returns_results(vector_store):
    """
    EN: Basic search sanity check.
    FR: Vérifie que search() renvoie bien (doc, score).
    """
    results = vector_store.search("revenue 2023", k=3)

    assert 0 < len(results) <= 3

    doc, score = results[0]
    assert hasattr(doc, "page_content")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_stats(vector_store):
    """
    EN: Stats must include total_docs.
    FR: Test simple pour stats().
    """
    stats = vector_store.stats()
    assert "total_docs" in stats
    assert stats["total_docs"] > 0
