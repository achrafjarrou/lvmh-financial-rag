import json
from typing import List, Dict, Any
from pathlib import Path

from src.rag_pipeline import RAGPipeline


def keyword_match_score(answer: str, keywords: List[str]) -> float:
    """
    Returns a simple keyword coverage score in [0, 1].

    This is intentionally simple:
    - it's cheap to compute
    - it gives a quick signal when the system is totally off
    - you can later replace it with more serious eval (exact match, citations, LLM-as-judge, etc.)
    """
    answer_lower = (answer or "").lower()
    if not keywords:
        return 0.0

    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords)


def evaluate_rag(dataset_path: Path) -> Dict[str, Any]:
    """
    Runs the RAG system on a golden dataset and returns aggregated metrics.

    Output includes:
    - overall keyword match
    - average latency
    - breakdown by category
    - breakdown by difficulty
    - per-question details
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Load dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Init RAG (no cache during evaluation)
    rag = RAGPipeline()

    results = []
    total = len(dataset)

    print(f"Running evaluation on {total} questions...\n")

    for idx, item in enumerate(dataset, start=1):
        question = item["question"]
        expected_kw = item.get("expected_keywords", [])

        # Query the system
        output = rag.query(question, use_cache=False)

        # Basic metrics
        km = keyword_match_score(output.get("answer", ""), expected_kw)
        latency = output.get("latency_ms", 0)

        results.append(
            {
                "id": item.get("id", idx),
                "question": question,
                "answer": output.get("answer", ""),
                "category": item.get("category", "unknown"),
                "difficulty": item.get("difficulty", "unknown"),
                "keyword_match": km,
                "latency_ms": latency,
            }
        )

        print(f"  [{idx}/{total}] {km:.0%} - {question[:70]}")

    # Overall aggregates
    avg_kw = sum(r["keyword_match"] for r in results) / total if total else 0.0
    avg_lat = sum(r["latency_ms"] for r in results) / total if total else 0.0

    # Breakdown by category
    by_category: Dict[str, Dict[str, Any]] = {}
    categories = sorted(set(r["category"] for r in results))
    for cat in categories:
        cat_items = [r for r in results if r["category"] == cat]
        by_category[cat] = {
            "count": len(cat_items),
            "avg_keyword_match": sum(r["keyword_match"] for r in cat_items) / len(cat_items),
            "avg_latency_ms": round(sum(r["latency_ms"] for r in cat_items) / len(cat_items), 2),
        }

    # Breakdown by difficulty
    by_difficulty: Dict[str, Dict[str, Any]] = {}
    difficulties = sorted(set(r["difficulty"] for r in results))
    for diff in difficulties:
        diff_items = [r for r in results if r["difficulty"] == diff]
        by_difficulty[diff] = {
            "count": len(diff_items),
            "avg_keyword_match": sum(r["keyword_match"] for r in diff_items) / len(diff_items),
            "avg_latency_ms": round(sum(r["latency_ms"] for r in diff_items) / len(diff_items), 2),
        }

    return {
        "total_questions": total,
        "avg_keyword_match": round(avg_kw, 3),
        "avg_latency_ms": round(avg_lat, 2),
        "by_category": by_category,
        "by_difficulty": by_difficulty,
        "details": results,
    }


if __name__ == "__main__":
    dataset_path = Path("evaluation/golden_dataset.json")

    print("=" * 70)
    print("RAG SYSTEM EVALUATION")
    print("=" * 70)

    metrics = evaluate_rag(dataset_path)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Questions: {metrics['total_questions']}")
    print(f"Avg keyword match: {metrics['avg_keyword_match']:.1%}")
    print(f"Avg latency: {metrics['avg_latency_ms']:.0f} ms")

    print("\nBy category:")
    for cat, stats in metrics["by_category"].items():
        print(f"  - {cat}: {stats['avg_keyword_match']:.1%} ({stats['count']} questions)")

    print("\nBy difficulty:")
    for diff, stats in metrics["by_difficulty"].items():
        print(f"  - {diff}: {stats['avg_keyword_match']:.1%} ({stats['count']} questions)")

    # Save results
    out_path = Path("evaluation/results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ Saved results to {out_path}")
