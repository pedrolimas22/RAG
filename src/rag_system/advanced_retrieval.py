"""
Advanced retrieval strategies: Multi-Query and Reranking.
"""

import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


class MultiQueryRetriever:
    """
    Generates multiple query variations to improve retrieval coverage.
    
    This addresses the limitation of single-query retrieval where the exact
    wording of the question might miss relevant documents.
    """
    
    def __init__(self, llm, num_queries: int = 3):
        """
        Initialize multi-query retriever.
        
        Args:
            llm: Language model for query generation
            num_queries: Number of query variations to generate
        """
        self.llm = llm
        self.num_queries = num_queries
        self.prompt = ChatPromptTemplate.from_template(
            """You are an AI assistant helping to improve search queries.
Given the original question, generate {num_queries} alternative versions that capture different aspects or phrasings of the same information need.

Original question: {question}

Generate {num_queries} alternative search queries (one per line):"""
        )
        self.output_parser = StrOutputParser()
    
    def generate_queries(self, original_query: str) -> List[str]:
        """
        Generate multiple query variations.
        
        Args:
            original_query: The user's original question
            
        Returns:
            List of query variations (includes original)
        """
        logger.info(f"Generating {self.num_queries} query variations...")
        
        try:
            # Generate alternative queries
            chain = self.prompt | self.llm | self.output_parser
            result = chain.invoke({
                "question": original_query,
                "num_queries": self.num_queries
            })
            
            # Parse queries from response
            queries = [q.strip() for q in result.split('\n') if q.strip()]
            
            # Always include original query
            all_queries = [original_query] + queries[:self.num_queries]
            
            logger.info(f"Generated queries: {all_queries}")
            return all_queries
            
        except Exception as e:
            logger.warning(f"Query generation failed: {e}. Using original query only.")
            return [original_query]


class DocumentReranker:
    """
    Reranks retrieved documents based on relevance to the query.
    
    This improves retrieval quality by re-scoring documents with a more
    sophisticated relevance model.
    """
    
    def __init__(self, llm):
        """
        Initialize document reranker.
        
        Args:
            llm: Language model for relevance scoring
        """
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(
            """You are a relevance scoring assistant. Given a query and a document, rate how relevant the document is to answering the query.

Query: {query}

Document:
{document}

Rate the relevance on a scale of 0-10 where:
- 0 = Completely irrelevant
- 5 = Somewhat relevant
- 10 = Highly relevant and directly answers the query

Respond with ONLY a single number between 0 and 10:"""
        )
        self.output_parser = StrOutputParser()
    
    def score_document(self, query: str, document: Document) -> float:
        """
        Score a single document's relevance to the query.
        
        Args:
            query: Search query
            document: Document to score
            
        Returns:
            Relevance score (0-10)
        """
        try:
            chain = self.prompt | self.llm | self.output_parser
            result = chain.invoke({
                "query": query,
                "document": document.page_content[:1000]  # Limit length
            })
            
            # Extract numeric score
            score_str = result.strip().split('\n')[0].strip()
            score = float(score_str)
            return max(0.0, min(10.0, score))  # Clamp to 0-10
            
        except Exception as e:
            logger.warning(f"Scoring failed for document: {e}")
            return 5.0  # Default medium score
    
    def rerank_documents(
        self, 
        query: str, 
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Document]:
        """
        Rerank documents by relevance to query.
        
        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Number of top documents to return (None = all)
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            return documents
        
        logger.info(f"Reranking {len(documents)} documents...")
        
        # Score each document
        scored_docs = []
        for doc in documents:
            score = self.score_document(query, doc)
            scored_docs.append((score, doc))
        
        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Return top-k or all
        reranked = [doc for score, doc in scored_docs]
        if top_k:
            reranked = reranked[:top_k]
        
        logger.info(f"Reranking complete. Top scores: {[s for s, _ in scored_docs[:3]]}")
        return reranked


class AdvancedRetriever:
    """
    Combines multi-query and reranking for improved retrieval.
    """
    
    def __init__(
        self,
        vectorstore_manager,
        llm,
        use_multi_query: bool = False,
        use_reranking: bool = False,
        num_queries: int = 3,
        retrieval_k: int = 10
    ):
        """
        Initialize advanced retriever.
        
        Args:
            vectorstore_manager: Vector store manager
            llm: Language model
            use_multi_query: Enable multi-query retrieval
            use_reranking: Enable reranking
            num_queries: Number of query variations
            retrieval_k: Number of docs to retrieve before reranking
        """
        self.vectorstore_manager = vectorstore_manager
        self.llm = llm
        self.use_multi_query = use_multi_query
        self.use_reranking = use_reranking
        self.retrieval_k = retrieval_k
        
        # Initialize components
        if use_multi_query:
            self.multi_query_retriever = MultiQueryRetriever(llm, num_queries)
        
        if use_reranking:
            self.reranker = DocumentReranker(llm)
    
    def retrieve(self, query: str, top_k: int) -> List[Document]:
        """
        Retrieve documents using advanced strategies.
        
        Args:
            query: User's question
            top_k: Number of final documents to return
            
        Returns:
            List of relevant documents
        """
        logger.info(f"Advanced retrieval: multi_query={self.use_multi_query}, reranking={self.use_reranking}")
        
        # Step 1: Generate query variations (if enabled)
        if self.use_multi_query:
            queries = self.multi_query_retriever.generate_queries(query)
        else:
            queries = [query]
        
        # Step 2: Retrieve documents for all queries
        all_documents = []
        seen_content = set()
        
        for q in queries:
            # Retrieve more documents if reranking (we'll filter later)
            k = self.retrieval_k if self.use_reranking else top_k
            docs = self.vectorstore_manager.similarity_search(q, k=k)
            
            # Deduplicate by content
            for doc in docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_documents.append(doc)
        
        logger.info(f"Retrieved {len(all_documents)} unique documents from {len(queries)} queries")
        
        # Step 3: Rerank documents (if enabled)
        if self.use_reranking and len(all_documents) > top_k:
            all_documents = self.reranker.rerank_documents(query, all_documents, top_k=top_k)
        else:
            # Just take top_k
            all_documents = all_documents[:top_k]
        
        logger.info(f"Returning {len(all_documents)} final documents")
        return all_documents
