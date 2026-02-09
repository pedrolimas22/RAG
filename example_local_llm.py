#!/usr/bin/env python
"""
Example using local LLM on M1/M2/M3 chips with Ollama.
Completely FREE - no API costs!
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.rag_system import RAGSystem, RAGConfig


def main():
    """Run RAG with local LLM."""
    
    print("="*80)
    print("RAG System - Local LLM Example (M1/M2/M3)")
    print("="*80)
    print("\n💡 This example uses Ollama for FREE local inference!")
    print("   No OpenAI API costs, everything runs on your Mac.\n")
    
    # Configuration for local LLM
    config = RAGConfig(
        # Local LLM settings
        use_local_llm=True,              # Use local instead of OpenAI
        local_llm_model="llama3.2:3b",   # Fast, high-quality model
        
        # Free embeddings
        embedding_model="huggingface",   # Free, no API needed
        
        # Document settings
        chunk_size=800,
        chunk_overlap=150,
        top_k=3,
        
        # Version
        version="local_v1"
    )
    
    print("Configuration:")
    print(f"  LLM: {config.local_llm_model} (Local, FREE)")
    print(f"  Embeddings: {config.embedding_model} (FREE)")
    print(f"  Total Cost: $0.00 💰\n")
    
    # Initialize RAG system
    print("Initializing RAG system...")
    try:
        rag = RAGSystem(config)
        print("✅ System initialized\n")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        print("SETUP REQUIRED:")
        print("="*80)
        print("1. Install Ollama:")
        print("   Visit: https://ollama.ai")
        print("   Or run: brew install ollama")
        print()
        print("2. Start Ollama server:")
        print("   ollama serve")
        print()
        print("3. Pull the model (in another terminal):")
        print("   ollama pull llama3.2:3b")
        print()
        print("4. Re-run this script")
        print("="*80)
        return
    
    # Ingest documents
    document_path = "documents/Transactions.csv"
    
    if Path(document_path).exists():
        print(f"Ingesting: {document_path}")
        try:
            num_chunks = rag.ingest(document_path, clear_existing=True)
            print(f"✅ Ingested {num_chunks} chunks\n")
        except Exception as e:
            print(f"❌ Ingestion failed: {e}\n")
            return
    else:
        print(f"⚠️  Document not found: {document_path}")
        print("   Please add a document to test with.\n")
        return
    
    # Example queries
    questions = [
        "What is the most expensive transaction?",
        "How many transactions are there?",
        "What categories exist?",
    ]
    
    print("="*80)
    print("Running Example Queries (100% Local, $0 cost)")
    print("="*80)
    
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}] Question: {question}")
        print("-"*80)
        
        try:
            result = rag.query(question, verbose=False)
            print(f"Answer: {result['result']}")
            print(f"Sources: {len(result['source_documents'])} documents")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "="*80)
    print("✅ All queries completed - TOTAL COST: $0.00!")
    print("="*80)
    print("\n💡 Try other models:")
    print("   - ollama pull llama3.2:7b  (higher quality)")
    print("   - ollama pull phi3:mini    (Microsoft, efficient)")
    print("   - ollama pull gemma2:2b    (Google, fast)")
    print()


if __name__ == "__main__":
    main()
