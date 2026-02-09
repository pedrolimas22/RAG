#!/usr/bin/env python
"""
CLI script for querying the RAG system.
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_system import RAGSystem, RAGConfig


def main():
    """Main query script."""
    parser = argparse.ArgumentParser(
        description="Query the RAG system"
    )
    parser.add_argument(
        "query",
        type=str,
        help="Question to ask the RAG system"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of source documents to retrieve (default: 3)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed processing information"
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Display source documents"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v3",
        help="Version tag for the collection (default: v3)"
    )
    parser.add_argument(
        "--embedding",
        choices=["openai", "huggingface"],
        default="huggingface",
        help="Embedding model to use (default: huggingface)"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = RAGConfig(
        top_k=args.top_k,
        version=args.version,
        embedding_model=args.embedding
    )
    
    # Initialize RAG system
    if args.verbose:
        print(f"Initializing RAG system...")
    rag = RAGSystem(config)
    
    # Run query
    try:
        result = rag.query(args.query, verbose=args.verbose)
        
        # Display answer
        print("\n" + "="*80)
        print("ANSWER:")
        print("="*80)
        print(result["result"])
        
        # Display sources if requested
        if args.show_sources:
            print("\n" + "="*80)
            print("SOURCE DOCUMENTS:")
            print("="*80)
            for i, doc in enumerate(result["source_documents"], 1):
                source = doc.metadata.get("source", "Unknown")
                print(f"\n[Source {i}] {source}")
                print("-"*80)
                preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                print(preview)
        
        print("\n")
        
    except Exception as e:
        print(f"\n✗ Query failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
