#!/usr/bin/env python
"""
Interactive REPL for querying the RAG system.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_system import RAGSystem, RAGConfig


def main():
    """Interactive query interface."""
    print("="*80)
    print("RAG System - Interactive Query Interface")
    print("="*80)
    print("Type 'exit' or 'quit' to end the session")
    print("Type 'sources' to toggle source document display")
    print("="*80)
    
    # Initialize RAG system
    try:
        config = RAGConfig()
        rag = RAGSystem(config)
        print(f"\n✓ RAG system initialized (version: {config.version})")
        print(f"  Embedding model: {config.embedding_model}")
        if config.use_local_llm: 
            print(f"  LLM model: {config.llm_model}") 
        else:
            print(f"  Local LLM model: {config.local_llm_model} (FREE)")
        print(f"  Top-K: {config.top_k}")
    except Exception as e:
        print(f"\n✗ Failed to initialize RAG system: {str(e)}")
        sys.exit(1)
    
    show_sources = False
    
    print("\n" + "="*80)
    print("Ready for questions!")
    print("="*80 + "\n")
    
    while True:
        try:
            # Get user input
            query = input("\nYour question: ").strip()
            
            # Handle commands
            if query.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break
            
            if query.lower() == 'sources':
                show_sources = not show_sources
                print(f"\nSource display: {'ON' if show_sources else 'OFF'}")
                continue
            
            if not query:
                continue
            
            # Process query
            result = rag.query(query, verbose=False)
            
            # Display answer
            print("\n" + "-"*80)
            print("Answer:")
            print("-"*80)
            print(result["result"])
            
            # Display sources if enabled
            if show_sources:
                print("\n" + "-"*80)
                print("Sources:")
                print("-"*80)
                for i, doc in enumerate(result["source_documents"], 1):
                    source = doc.metadata.get("source", "Unknown")
                    print(f"\n[{i}] {source}")
                    preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    print(f"    {preview}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")


if __name__ == "__main__":
    main()
