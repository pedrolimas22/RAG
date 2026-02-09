"""
Main RAG system interface.
"""

import logging
from typing import Dict, Any, Optional, List
from langchain_core.documents import Document
from .config import RAGConfig
from .ingestion import IngestionPipeline
from .query import QueryPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Main interface for the RAG system.
    
    Combines ingestion and query pipelines into a unified interface.
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize RAG system.
        
        Args:
            config: RAG configuration (creates default if not provided)
        """
        self.config = config or RAGConfig()
        self.ingestion_pipeline = IngestionPipeline(self.config)
        self.query_pipeline = QueryPipeline(self.config)
        logger.info(f"RAG System initialized with version {self.config.version}")
    
    def ingest(self, file_path: Optional[str] = None, clear_existing: bool = False) -> int:
        """
        Ingest documents into the vector store.
        
        Args:
            file_path: Path to file to ingest
            clear_existing: Whether to clear existing documents
            
        Returns:
            Number of chunks added
        """
        return self.ingestion_pipeline.ingest(file_path, clear_existing)
    
    def query(self, question: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: User's question
            verbose: Whether to log detailed information
            
        Returns:
            Dictionary with 'result' and 'source_documents'
        """
        return self.query_pipeline.query(question, verbose)
    
    def search(self, query: str, k: int = None) -> List[Document]:
        """
        Search for similar documents without generating an answer.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of similar documents
        """
        return self.query_pipeline.get_similar_documents(query, k)
    
    def update_config(self, **kwargs) -> None:
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")
        
        # Reinitialize pipelines if needed
        if any(k in kwargs for k in ['embedding_model', 'collection_name', 'top_k']):
            logger.info("Reinitializing pipelines due to config change...")
            self.ingestion_pipeline = IngestionPipeline(self.config)
            self.query_pipeline = QueryPipeline(self.config)
