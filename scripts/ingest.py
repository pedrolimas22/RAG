#!/usr/bin/env python
"""
CLI script for ingesting documents into the RAG system.
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_system import RAGSystem, RAGConfig


def main():
    """Main ingestion script."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG system vector store"
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the document file to ingest"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing documents before ingesting"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Size of document chunks (default: 1000)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks (default: 200)"
    )
    parser.add_argument(
        "--embedding",
        choices=["openai", "huggingface"],
        default="huggingface",
        help="Embedding model to use (default: huggingface)"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v3",
        help="Version tag for the collection (default: v3)"
    )
    
    args = parser.parse_args()
    
    # Validate file exists
    if not Path(args.file_path).exists():
        print(f"Error: File not found: {args.file_path}")
        sys.exit(1)
    
    # Create configuration
    config = RAGConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding,
        version=args.version
    )
    
    # Initialize RAG system
    print(f"Initializing RAG system...")
    rag = RAGSystem(config)
    
    # Run ingestion
    try:
        num_chunks = rag.ingest(args.file_path, clear_existing=args.clear)
        print(f"\n✓ Successfully ingested {num_chunks} chunks!")
    except Exception as e:
        print(f"\n✗ Ingestion failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
