"""
RAG System - A modular Retrieval-Augmented Generation system.
"""

from .config import RAGConfig
from .rag import RAGSystem
from .local_llm import LocalLLM, create_llm
from .advanced_retrieval import AdvancedRetriever, MultiQueryRetriever, DocumentReranker

__version__ = "1.0.0"
__all__ = [
    "RAGConfig",
    "RAGSystem",
    "LocalLLM",
    "create_llm",
    "AdvancedRetriever",
    "MultiQueryRetriever",
    "DocumentReranker"
]
