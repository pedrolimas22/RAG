#!/usr/bin/env python
"""
Example demonstrating advanced retrieval: Multi-Query + Reranking.

This shows how to enable advanced retrieval features for better results.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.rag_system import RAGSystem, RAGConfig


def compare_retrieval_strategies():
    """Compare basic vs advanced retrieval."""
    
    question = "What is the most expensive transaction?"
    
    print("="*80)
    print("RAG System - Advanced Retrieval Comparison")
    print("="*80)
    print(f"\nTest Question: {question}\n")
    
    # Strategy 1: Basic Retrieval (Default)
    print("\n" + "="*80)
    print("STRATEGY 1: Basic Retrieval (Default)")
    print("="*80)
    
    config_basic = RAGConfig(
        use_local_llm=True,
        local_llm_model="llama3.2:3b",
        embedding_model="huggingface",
        top_k=3,
        use_multi_query=False,
        use_reranking=False
    )
    
    rag_basic = RAGSystem(config_basic)
    result_basic = rag_basic.query(question, verbose=True)
    
    print(f"\nAnswer: {result_basic['result']}\n")
    print(f"Sources used: {len(result_basic['source_documents'])}")
    
    # Strategy 2: Multi-Query Only
    print("\n" + "="*80)
    print("STRATEGY 2: Multi-Query Retrieval")
    print("="*80)
    
    config_multi = RAGConfig(
        use_local_llm=True,
        local_llm_model="llama3.2:3b",
        embedding_model="huggingface",
        top_k=3,
        use_multi_query=True,   # ✅ Generate query variations
        use_reranking=False,
        num_queries=3           # Generate 3 variations
    )
    
    rag_multi = RAGSystem(config_multi)
    result_multi = rag_multi.query(question, verbose=True)
    
    print(f"\nAnswer: {result_multi['result']}\n")
    print(f"Sources used: {len(result_multi['source_documents'])}")
    
    # Strategy 3: Multi-Query + Reranking (Best Quality)
    print("\n" + "="*80)
    print("STRATEGY 3: Multi-Query + Reranking (Best)")
    print("="*80)
    
    config_advanced = RAGConfig(
        use_local_llm=True,
        local_llm_model="llama3.2:3b",
        embedding_model="huggingface",
        top_k=3,
        use_multi_query=True,   # ✅ Generate query variations
        use_reranking=True,      # ✅ Rerank by relevance
        num_queries=3,
        retrieval_k=10          # Retrieve 10, return best 3 after reranking
    )
    
    rag_advanced = RAGSystem(config_advanced)
    result_advanced = rag_advanced.query(question, verbose=True)
    
    print(f"\nAnswer: {result_advanced['result']}\n")
    print(f"Sources used: {len(result_advanced['source_documents'])}")
    
    # Summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print("\nRetrieval Strategies:")
    print("  1. Basic:           Single query, direct retrieval (fastest)")
    print("  2. Multi-Query:     Multiple query perspectives (better coverage)")
    print("  3. Multi+Rerank:    Best results, reranked by relevance (best quality)")
    print("\nTrade-offs:")
    print("  Basic:              Fast, low cost, may miss relevant docs")
    print("  Multi-Query:        Medium speed, better coverage")
    print("  Multi+Rerank:       Slower, highest quality, finds most relevant docs")
    print()


def main():
    """Run advanced retrieval example."""
    
    print("🚀 Advanced Retrieval Features")
    print("="*80)
    print("\nThis example demonstrates:")
    print("  • Multi-Query: Generate query variations for better coverage")
    print("  • Reranking: Re-score documents by relevance")
    print("\n⚠️  Note: These features increase LLM calls but improve quality")
    print("="*80)
    
    # Check if documents are ingested
    from pathlib import Path
    if not Path("data/chroma_db").exists():
        print("\n❌ No vector database found!")
        print("Please ingest documents first:")
        print("  python scripts/ingest.py documents/Transactions.csv")
        return
    
    try:
        compare_retrieval_strategies()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. Ollama is running: ollama serve")
        print("  2. Model is downloaded: ollama pull llama3.2:3b")
        print("  3. Documents are ingested")


if __name__ == "__main__":
    main()
