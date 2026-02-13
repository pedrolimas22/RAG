"""
Document loading and preprocessing for RAG system.
"""

import logging
from typing import List
from pathlib import Path
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from .config import RAGConfig

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Handles document loading and chunking."""
    
    def __init__(self, config: RAGConfig):
        """
        Initialize document loader.
        
        Args:
            config: RAG configuration
        """
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
        )
    
    def load_documents(self, file_path: str = None) -> List[Document]:
        """
        Load documents from file or directory.
        
        Args:
            file_path: Path to file or directory (defaults to config.documents_path)
            
        Returns:
            List of loaded documents
        """
        path = file_path or self.config.documents_path
        logger.info(f"Loading documents from: {path}")
        
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        
        if path_obj.is_file():
            documents = self._load_single_file(str(path_obj))
        else:
            raise ValueError("Directory loading not implemented. Please provide a file path.")
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def _load_single_file(self, file_path: str) -> List[Document]:
        """Load a single file based on its extension."""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension == ".csv":
            return self._load_csv(file_path)
        elif extension in [".txt", ".md"]:
            # Handle text and markdown files the same way
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return [Document(page_content=content, metadata={"source": file_path})]
        elif extension == ".pdf":
            return self._load_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}. Supported types: .pdf, .csv, .txt, .md")
    
    def _load_csv(self, file_path: str) -> List[Document]:
        """Load CSV file."""
        loader = CSVLoader(
            file_path=file_path,
            csv_args={
                "delimiter": ",",
                "quotechar": '"'
            }
        )
        return loader.load()
    
    def _load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF file."""
        loader = PyPDFLoader(file_path)

        return loader.load()
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: Documents to split
            
        Returns:
            List of document chunks
        """
        logger.info(f"Splitting {len(documents)} documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
