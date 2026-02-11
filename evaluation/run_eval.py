from pathlib import Path
from evaluation.metrics import evaluate_rag


def main():
    dataset_path = Path("evaluation/golden_dataset.json")

    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        raise SystemExit(1)

    print("Starting evaluation...\n")

    metrics = evaluate_rag(dataset_path)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\nAvg keyword match: {metrics['avg_keyword_match']:.1%}")
    print(f"Avg latency: {metrics['avg_latency_ms']:.0f} ms")

    print("\n✓ Evaluation completed successfully")


if __name__ == "__main__":
    main()
