"""
Vector store management for RAG system.
"""

import logging
from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document
from .config import RAGConfig
from .embeddings import create_embeddings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages ChromaDB vector store operations."""
    
    def __init__(self, config: RAGConfig):
        """
        Initialize vector store manager.
        
        Args:
            config: RAG configuration
        """
        self.config = config
        self.embeddings = create_embeddings(config)
        self.vectorstore = self._create_vectorstore()
    
    def _create_vectorstore(self) -> Chroma:
        """Create and return ChromaDB vectorstore."""
        logger.info(f"Initializing vector store with collection: {self.config.collection_name}")
        
        if self.config.use_chroma_cloud:
            return Chroma(
                embedding_function=self.embeddings,
                collection_name=self.config.collection_name,
                chroma_cloud_api_key=self.config.chroma_api_key,
                tenant=self.config.chroma_tenant,
                database=self.config.database_name
            )
        else:
            return Chroma(
                embedding_function=self.embeddings,
                collection_name=self.config.collection_name,
                persist_directory=self.config.persist_directory
            )
    
    def add_documents(self, documents: List[Document], batch_size: int = 300) -> None:
        """
        Add documents to vector store in batches.
        
        Args:
            documents: List of documents to add
            batch_size: Number of documents per batch
        """
        total_batches = (len(documents) + batch_size - 1) // batch_size
        logger.info(f"Adding {len(documents)} documents in {total_batches} batches...")
        
        for i in range(0, len(documents), batch_size):
            batch_chunks = documents[i:i + batch_size]
            batch_ids = [f"doc_{i}_{j}" for j in range(len(batch_chunks))]
            
            self.vectorstore.add_documents(
                documents=batch_chunks,
                ids=batch_ids
            )
            
            batch_num = i // batch_size + 1
            logger.info(f"  Added batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)")
    
    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """
        Perform similarity search on vector store.
        
        Args:
            query: Search query
            k: Number of results to return (defaults to config.top_k)
            
        Returns:
            List of relevant documents
        """
        k = k or self.config.top_k
        logger.info(f"Performing similarity search with k={k}")
        return self.vectorstore.similarity_search(query, k=k)
    
    def clear_collection(self) -> None:
        """Clear all documents from the current collection."""
        logger.warning(f"Clearing collection: {self.config.collection_name}")
        # ChromaDB doesn't have a direct clear method, so we delete and recreate
        self.vectorstore = self._create_vectorstore()
