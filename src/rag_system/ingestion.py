"""
Document ingestion pipeline for RAG system.
"""

import logging
from typing import Optional
from .config import RAGConfig
from .document_loader import DocumentLoader
from .vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Handles the complete document ingestion process."""
    
    def __init__(self, config: RAGConfig):
        """
        Initialize ingestion pipeline.
        
        Args:
            config: RAG configuration
        """
        self.config = config
        self.document_loader = DocumentLoader(config)
        self.vectorstore_manager = VectorStoreManager(config)
    
    def ingest(self, file_path: Optional[str] = None, clear_existing: bool = False) -> int:
        """
        Run the complete ingestion pipeline.
        
        Args:
            file_path: Path to file to ingest (defaults to config.documents_path)
            clear_existing: Whether to clear existing documents first
            
        Returns:
            Number of chunks added to vector store
        """
        logger.info("="*80)
        logger.info(f"STARTING INGESTION PIPELINE - VERSION {self.config.version}")
        logger.info("="*80)
        
        try:
            # Step 1: Load documents
            logger.info("\n[1/3] Loading documents...")
            documents = self.document_loader.load_documents(file_path)
            logger.info(f"✓ Loaded {len(documents)} documents")
            
            # Step 2: Chunk documents
            logger.info("\n[2/3] Chunking documents...")
            chunks = self.document_loader.split_documents(documents)
            logger.info(f"✓ Created {len(chunks)} chunks")
            
            # Step 3: Add to vector store
            logger.info(f"\n[3/3] Adding to vector store...")
            logger.info(f"  Collection: {self.config.collection_name}")
            
            if clear_existing:
                self.vectorstore_manager.clear_collection()
            
            self.vectorstore_manager.add_documents(chunks)
            logger.info(f"✓ Successfully added {len(chunks)} chunks to vector store")
            
            logger.info("\n" + "="*80)
            logger.info("INGESTION COMPLETE")
            logger.info("="*80)
            
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Ingestion failed: {str(e)}", exc_info=True)
            raise
