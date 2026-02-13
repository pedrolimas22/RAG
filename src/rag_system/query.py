"""
Query/inference pipeline for RAG system.
"""

import logging
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from .config import RAGConfig
from .vectorstore import VectorStoreManager
from .local_llm import create_llm
from .advanced_retrieval import AdvancedRetriever

logger = logging.getLogger(__name__)


class QueryPipeline:
    """Handles query processing and answer generation."""
    
    def __init__(self, config: RAGConfig):
        """
        Initialize query pipeline.
        
        Args:
            config: RAG configuration
        """
        self.config = config
        self.vectorstore_manager = VectorStoreManager(config)
        self.llm = create_llm(config)  # Supports both OpenAI and local LLM
        self.output_parser = StrOutputParser()
        self._setup_prompt_template()
        
        # Initialize advanced retrieval if enabled
        if config.use_multi_query or config.use_reranking:
            logger.info("Advanced retrieval enabled: multi_query={}, reranking={}".format(
                config.use_multi_query, config.use_reranking
            ))
            self.advanced_retriever = AdvancedRetriever(
                vectorstore_manager=self.vectorstore_manager,
                llm=self.llm,
                use_multi_query=config.use_multi_query,
                use_reranking=config.use_reranking,
                num_queries=config.num_queries,
                retrieval_k=config.retrieval_k
            )
        else:
            self.advanced_retriever = None
    
    def _setup_prompt_template(self):
        """Create the prompt template for the RAG chain."""
        self.prompt_template = ChatPromptTemplate.from_template(
            """Based on the following documents, answer the question clearly and concisely (max of 2 or 3 paragraphs).
If the answer is not in the documents, say so and don't use any other information.

Documents: {context}

Question: {query}

Answer:"""
        )
    
    def query(self, question: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Process a query and generate an answer.
        
        Args:
            question: User's question
            verbose: Whether to log detailed information
            
        Returns:
            Dictionary containing 'result' and 'source_documents'
        """
        if verbose:
            logger.info("="*80)
            logger.info(f"RUNNING INFERENCE - VERSION {self.config.version}")
            logger.info("="*80)
        
        try:
            # Step 1: Retrieve documents (basic or advanced)
            if verbose:
                logger.info(f"\n[1/5] Retrieving relevant documents...")
                logger.info(f"  Query: '{question}'")
            
            # Use advanced retrieval if enabled, otherwise basic similarity search
            if self.advanced_retriever:
                results = self.advanced_retriever.retrieve(question, top_k=self.config.top_k)
            else:
                results = self.vectorstore_manager.similarity_search(question)
            
            if verbose:
                logger.info(f"✓ Retrieved {len(results)} relevant chunks")
            
            # Step 2: Format context
            if verbose:
                logger.info(f"\n[2/5] Formatting context for LLM...")
            
            context = "\n\n".join([doc.page_content for doc in results])
            
            if verbose:
                logger.info(f"✓ Context formatted ({len(context)} characters)")
            
            # Step 3: Create chain
            if verbose:
                logger.info("\n[3/5] Creating chain...")
            
            chain = self.prompt_template | self.llm | self.output_parser
            
            if verbose:
                logger.info("✓ Chain created: prompt → llm → parser")
            
            # Step 4: Generate answer
            if verbose:
                logger.info(f"\n[4/5] Generating answer...")
            
            answer = chain.invoke({
                "context": context,
                "query": question
            })
            
            if verbose:
                logger.info(f"✓ Answer generated ({len(answer)} characters)")
                logger.info("\n" + "="*80)
                logger.info("INFERENCE COMPLETE")
                logger.info("="*80)
            
            return {
                "result": answer,
                "source_documents": results
            }
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}", exc_info=True)
            raise
    
    def get_similar_documents(self, query: str, k: int = None) -> List[Document]:
        """
        Get similar documents without generating an answer.
        
        Args:
            query: Search query
            k: Number of results (defaults to config.top_k)
            
        Returns:
            List of similar documents
        """
        return self.vectorstore_manager.similarity_search(query, k=k)
