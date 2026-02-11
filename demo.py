
from src.rag_pipeline import RAGPipeline

def main():
    print("=" * 70)
    print("🤖 LVMH FINANCIAL RAG - DEMO")
    print("=" * 70)
    
    # Init
    rag = RAGPipeline()
    
    # Questions prédéfinies
    questions = [
        "Quel est le chiffre d'affaires de LVMH en 2023?",
        "Combien de magasins LVMH possède-t-il?",
        "Quelle est la marge opérationnelle en 2023?",
        "Quels sont les principaux marchés géographiques?",
        "Comparez les performances 2022 vs 2023"
    ]
    
    print(f"\nTest sur {len(questions)} questions:\n")
    
    for i, q in enumerate(questions, 1):
        print(f"\n{'─' * 70}")
        print(f"[{i}/{len(questions)}] {q}")
        print("─" * 70)
        
        result = rag.query(q)
        
        print(f"\n✓ Réponse ({result['latency_ms']}ms, cached={result['from_cache']}):")
        print(f"  {result['answer']}")
        
        if result.get('sources'):
            print(f"\n📄 Sources:")
            for src in result['sources'][:3]:
                print(f"  • Page {src['page']} (score: {src['score']:.3f})")
    
    # Métriques
    print(f"\n{'=' * 70}")
    print("📊 MÉTRIQUES SYSTÈME")
    print("=" * 70)
    
    metrics = rag.get_metrics()
    print(f"Queries totales: {metrics['total_queries']}")
    print(f"Cache hits: {metrics['cache_hits']} ({metrics['cache_hit_rate']:.1%})")
    print(f"Latence moyenne: {metrics['avg_latency_ms']:.0f}ms")
    print(f"Documents en DB: {metrics['db_stats']['total_docs']}")
    
    print(f"\n{'=' * 70}")
    print("✓ Démo terminée")
    print("=" * 70)

if __name__ == "__main__":
    main()