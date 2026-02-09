"""
Embedding model management for RAG system.
"""

import logging
from typing import Union
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from .config import RAGConfig

logger = logging.getLogger(__name__)


def create_embeddings(config: RAGConfig) -> Union[OpenAIEmbeddings, HuggingFaceEmbeddings]:
    """
    Create embeddings model based on configuration.
    
    Args:
        config: RAG configuration
        
    Returns:
        Embeddings model instance
    """
    logger.info(f"Creating {config.embedding_model} embeddings model...")
    
    if config.embedding_model == "openai":
        return OpenAIEmbeddings(
            model=config.openai_embedding_model,
            openai_api_key=config.openai_api_key
        )
    elif config.embedding_model == "huggingface":
        return HuggingFaceEmbeddings(
            model_name=config.huggingface_model
        )
    else:
        raise ValueError(f"Unsupported embedding model: {config.embedding_model}")
