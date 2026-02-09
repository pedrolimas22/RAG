"""
RAG System - A modular Retrieval-Augmented Generation system.
"""

from .config import RAGConfig
from .rag import RAGSystem
from .local_llm import LocalLLM, create_llm

__version__ = "1.0.0"
__all__ = ["RAGConfig", "RAGSystem", "LocalLLM", "create_llm"]
