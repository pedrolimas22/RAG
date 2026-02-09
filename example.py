#!/usr/bin/env python
"""
Example script demonstrating how to use the RAG system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_system import RAGSystem, RAGConfig


def main():
    """Run example RAG workflow."""
    print("="*80)
    print("RAG System - Example Usage")
    print("="*80)
    
    # 1. Create configuration
    print("\n1. Creating configuration...")
    config = RAGConfig(
        chunk_size=1000,
        chunk_overlap=200,
        embedding_model="huggingface",  # Free, no API key needed     
        llm_model="gpt-4o-mini",
        version="example_v1"
    )
    print(f"   ✓ Config created with version: {config.version}")
    
    # 2. Initialize RAG system
    print("\n2. Initializing RAG system...")
    rag = RAGSystem(config)
    print("   ✓ System initialized")
    
    # 3. Ingest documents
    print("\n3. Ingesting documents...")
    document_path = "documents/Transactions.csv"
    
    if not Path(document_path).exists():
        print(f"   ⚠ Document not found: {document_path}")
        print("   Please add a document to the documents/ directory")
        return
    
    try:
        num_chunks = rag.ingest(document_path, clear_existing=True)
        print(f"   ✓ Ingested {num_chunks} chunks")
    except Exception as e:
        print(f"   ✗ Ingestion failed: {e}")
        return
    
    # 4. Query the system
    print("\n4. Querying the system...")
    questions = [
        "What is the most expensive transaction?",
        "How many transactions are there?",
        "What categories exist in the data?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n   Question {i}: {question}")
        try:
            result = rag.query(question, verbose=False)
            print(f"   Answer: {result['result'][:200]}...")
            print(f"   Sources: {len(result['source_documents'])} documents")
        except Exception as e:
            print(f"   ✗ Query failed: {e}")
    
    print("\n" + "="*80)
    print("Example completed!")
    print("="*80)
    print("\nNext steps:")
    print("  • Try the interactive mode: python scripts/interactive.py")
    print("  • Launch the web UI: python web/app.py")
    print("  • Read the docs: cat README.md")
    print()


if __name__ == "__main__":
    main()
