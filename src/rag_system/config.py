"""
Configuration management for RAG system.
"""

import os
from dataclasses import dataclass, field
from typing import Literal, Optional
from pathlib import Path
from dotenv import load_dotenv


@dataclass
class RAGConfig:
    """Configuration class for RAG system."""
    
    # Document processing
    chunk_size: int = 800  # Reduced: smaller chunks = less context per retrieval
    chunk_overlap: int = 150  # Reduced: less overlap = fewer total chunks
    file_type: Literal["txt", "csv", "pdf"] = "csv"
    
    # Embedding configuration
    embedding_model: Literal["openai", "huggingface"] = "huggingface"  # FREE!
    openai_embedding_model: str = "text-embedding-3-small"
    huggingface_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # LLM configuration
    use_local_llm: bool = True  # Set to True for FREE local inference on M1/M2/M3
    llm_model: str = "gpt-4o-mini"  # OpenAI model (if use_local_llm=False)
    local_llm_model: str = "llama3.2:3b"  # Ollama model (if use_local_llm=True)
    temperature: float = 0.0  # Doesn't affect cost
    
    # Retrieval configuration
    top_k: int = 2  # REDUCED from 3: Fewer docs = less tokens to LLM
    
    # Vector store configuration
    collection_name: str = "rag_documents"
    version: str = "v3"
    use_chroma_cloud: bool = True
    database_name: str = "code_for_all_rag"
    
    # Paths
    documents_path: str = "./documents"
    persist_directory: str = "./chroma_db"
    
    # API Keys (loaded from environment)
    openai_api_key: Optional[str] = field(default=None, repr=False)
    chroma_api_key: Optional[str] = field(default=None, repr=False)
    chroma_tenant: Optional[str] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Load environment variables and validate configuration."""
        # Load .env file from project root (parent of src directory)
        # This ensures .env is found regardless of where the script is run from
        config_file = Path(__file__).resolve()
        project_root = config_file.parent.parent.parent
        env_path = project_root / '.env'
        
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
            print(f"✓ Loaded .env from: {env_path}")
        else:
            load_dotenv()  # Try to load from environment
            print(f"⚠ No .env file found at {env_path}, using environment variables")
        
        # Load API keys from environment if not provided
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if self.chroma_api_key is None:
            self.chroma_api_key = os.getenv("CHROMA_API_KEY")
        
        if self.chroma_tenant is None:
            self.chroma_tenant = os.getenv("CHROMA_TENANT")
        
        # Validate required keys (only if using OpenAI)
        if not self.use_local_llm and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found. Set use_local_llm=True to use local LLM instead.")
        
        if self.use_chroma_cloud:
            if not self.chroma_api_key:
                raise ValueError("CHROMA_API_KEY required for ChromaDB Cloud")
            if not self.chroma_tenant:
                raise ValueError("CHROMA_TENANT required for ChromaDB Cloud")
        
        # Update collection name with version
        if self.version:
            self.collection_name = f"{self.collection_name}_{self.version}"
    
    def get_embedding_model_name(self) -> str:
        """Get the appropriate embedding model name based on configuration."""
        if self.embedding_model == "openai":
            return self.openai_embedding_model
        else:
            return self.huggingface_model
