"""
Local LLM support for M1/M2/M3 Apple Silicon chips.
Uses Ollama for efficient local inference with Metal acceleration.
"""

import logging
from typing import Optional
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class LocalLLM:
    """
    Manages local LLM inference optimized for Apple Silicon (M1/M2/M3).
    
    Uses Ollama which provides:
    - Metal GPU acceleration for M1/M2/M3 chips
    - Fast inference
    - No API costs
    - Privacy (all processing local)
    
    Prerequisites:
        Install Ollama: https://ollama.ai
        Pull a model: `ollama pull llama3.2:3b`
    """
    
    RECOMMENDED_MODELS = {
        "small": "llama3.2:3b",      # 3B params, very fast, good quality
        "medium": "llama3.2:7b",     # 7B params, balanced speed/quality
        "large": "llama3.1:8b",      # 8B params, high quality
        "phi": "phi3:mini",          # Microsoft Phi-3, efficient
        "gemma": "gemma2:2b",        # Google Gemma, small and fast
    }
    
    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        num_ctx: int = 4096,  # Context window size
    ):
        """
        Initialize local LLM.
        
        Args:
            model_name: Ollama model name (e.g., "llama3.2:3b")
            base_url: Ollama server URL
            temperature: LLM temperature (0 = deterministic)
            num_ctx: Context window size in tokens
        """
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.num_ctx = num_ctx
        
        logger.info(f"Initializing local LLM: {model_name}")
        self._validate_ollama_installed()
        self.llm = self._create_llm()
    
    def _validate_ollama_installed(self):
        """Check if Ollama is installed and running."""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                logger.info("✅ Ollama server is running")
                
                # Check if model is available
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                if self.model_name not in model_names:
                    logger.warning(f"⚠️  Model '{self.model_name}' not found")
                    logger.warning(f"   Available models: {', '.join(model_names)}")
                    logger.warning(f"   Run: ollama pull {self.model_name}")
                else:
                    logger.info(f"✅ Model '{self.model_name}' is available")
            else:
                raise ConnectionError("Ollama server returned non-200 status")
                
        except Exception as e:
            logger.error(f"❌ Ollama not accessible: {e}")
            logger.error("\n" + "="*80)
            logger.error("OLLAMA SETUP REQUIRED:")
            logger.error("="*80)
            logger.error("1. Install Ollama:")
            logger.error("   Visit: https://ollama.ai")
            logger.error("   Or: brew install ollama")
            logger.error("")
            logger.error("2. Start Ollama:")
            logger.error("   ollama serve")
            logger.error("")
            logger.error("3. Pull a model:")
            logger.error(f"   ollama pull {self.model_name}")
            logger.error("="*80)
            raise
    
    def _create_llm(self) -> Ollama:
        """Create Ollama LLM instance."""
        return Ollama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature,
            num_ctx=self.num_ctx,
        )
    
    def get_llm(self) -> Ollama:
        """Get the LLM instance."""
        return self.llm
    
    @classmethod
    def list_available_models(cls, base_url: str = "http://localhost:11434") -> list:
        """
        List all locally available Ollama models.
        
        Returns:
            List of model names
        """
        try:
            import requests
            response = requests.get(f"{base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [m['name'] for m in models]
            return []
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            return []
    
    @classmethod
    def get_recommended_model(cls, size: str = "small") -> str:
        """
        Get recommended model name for given size.
        
        Args:
            size: One of "small", "medium", "large", "phi", "gemma"
            
        Returns:
            Model name
        """
        return cls.RECOMMENDED_MODELS.get(size, cls.RECOMMENDED_MODELS["small"])


def create_llm(config):
    """
    Factory function to create LLM based on configuration.
    
    Args:
        config: RAGConfig instance
        
    Returns:
        LLM instance (OpenAI or Local)
    """
    if hasattr(config, 'use_local_llm') and config.use_local_llm:
        logger.info("Using local LLM (Ollama)")
        local_model = getattr(config, 'local_llm_model', 'llama3.2:3b')
        local_llm = LocalLLM(
            model_name=local_model,
            temperature=config.temperature
        )
        return local_llm.get_llm()
    else:
        logger.info(f"Using OpenAI LLM: {config.llm_model}")
        return ChatOpenAI(
            model=config.llm_model,
            temperature=config.temperature,
            api_key=config.openai_api_key
        )
