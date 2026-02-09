"""
Simple RAG (Retrieval-Augmented Generation) System using LangChain
This system loads documents, creates embeddings, stores them in a vector database,
and allows you to query the documents using natural language.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleRAG:
    def __init__(
        self, 
        documents_path: str = "./documents", 
        persist_directory: str = "./chroma_db",
        config: Optional[RAGConfig] = None
    ):
        """
        Initialize the RAG system.
        
        Args:
            documents_path: Path to the directory containing documents
            persist_directory: Path to store the vector database
            config: RAGConfig instance for system settings
        """
        self.documents_path = documents_path
        self.persist_directory = persist_directory
        self.config = config or RAGConfig()
        self.vectorstore: Optional[Chroma] = None
        self.qa_chain: Optional[RetrievalQA] = None
        
        # Check for OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    def load_documents(self, file_type: Optional[str] = None) -> List[Document]:
        """Load documents from the specified directory."""
        file_type = file_type or self.config.file_type
        logger.info(f"Loading {file_type} documents from {self.documents_path}...")
        
        # Load files
        if file_type == "txt":
            loader = DirectoryLoader(
                self.documents_path,
                glob="**/*.txt",
                loader_cls=TextLoader
            )
        elif file_type == "csv":
            loader = DirectoryLoader(
                self.documents_path,
                glob="**/*.csv",
                loader_cls=CSVLoader(
                    encoding="utf-8",
                    csv_args={"delimiter": ",", "quotechar": '"', "fieldnames": ["DATE", "OUTFLOW", "INFLOW", "CATEGORY", "ACCOUNT", "MEMO", "STATUS"]}                    
                )
            )
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks for better retrieval."""
        logger.info("Splitting documents into chunks...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def create_vectorstore(self, chunks: List[Document]) -> Chroma:
        """Create a vector store from document chunks."""
        logger.info("Creating vector store...")
        
        try:
            embeddings = OpenAIEmbeddings(
                model=self.config.embedding_model
            )
            
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=self.persist_directory
            )
            
            logger.info("Vector store created and persisted")
            return self.vectorstore
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def load_existing_vectorstore(self) -> Chroma:
        """Load an existing vector store from disk."""
        logger.info("Loading existing vector store...")
        
        embeddings = OpenAIEmbeddings(model=self.config.embedding_model)
        
        # Check if using Chroma Cloud or local
        if os.getenv("CHROMA_API_KEY"):
            collection_name = "csv_docs_transaction"
            self.vectorstore = Chroma(
                embedding_function=embeddings,
                collection_name=collection_name,
                chroma_cloud_api_key=os.getenv("CHROMA_API_KEY"),
                tenant=os.getenv("CHROMA_TENANT"),
                database="code_for_all_rag"
            )
        else:
            # Local Chroma DB
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embeddings
            )
        
        logger.info("Vector store loaded")
        return self.vectorstore
    
def setup_qa_chain(self) -> RetrievalQA:
        """Set up the question-answering chain."""
        logger.info("Setting up QA chain...")
        
        # Create a custom prompt template
        template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

Context: {context}

Question: {question}

Helpful Answer:"""
        
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )
        
        # Create the LLM
        llm = ChatOpenAI(
            model_name=self.config.llm_model, 
            temperature=self.config.temperature
        )
        
        # Create the retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": self.config.top_k}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        
        logger.info("QA chain ready")
        return self.qa_chain
    
    def initialize(self, force_rebuild: bool = False) -> None:
        """
        Initialize the RAG system.
        
        Args:
            force_rebuild: If True, rebuild the vector store from scratch
        """
        try:
            # Check if vector store exists
            vectorstore_exists = os.path.exists(self.persist_directory)
            
            if force_rebuild or not vectorstore_exists:
                # Load and process documents
                documents = self.load_documents()
                chunks = self.split_documents(documents)
                self.create_vectorstore(chunks)
            else:
                # Load existing vector store
                self.load_existing_vectorstore()
            
            # Setup QA chain
            self.setup_qa_chain()
            logger.info("RAG system initialized successfully!")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary with 'result' and 'source_documents'
        """
        if not self.qa_chain:
            raise ValueError("RAG system not initialized. Call initialize() first.")
        
        logger.info(f"Processing query: {question}")
        
        try:
            response = self.qa_chain({"query": question})
            logger.info(f"Found {len(response['source_documents'])} relevant documents")
            return response
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise


def main():
    """Main function to demonstrate RAG system usage."""
    
    # Create RAG instance
    rag = SimpleRAG(documents_path="./documents")
    
    # Initialize the system (this will load or create the vector store)
    rag.initialize(force_rebuild=False)
    
    # Example queries
    questions = [
        "What is the main topic of these documents?",
        "Can you summarize the key points?",
    ]
    
    print("\n" + "="*50)
    print("Running example queries...")
    print("="*50)
    
    for question in questions:
        response = rag.query(question)
        print("\n" + "-"*50 + "\n")
    
    # Interactive mode
    print("\n" + "="*50)
    print("Interactive Mode - Type 'quit' to exit")
    print("="*50)
    
    while True:
        user_question = input("\nYour question: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if user_question:
            rag.query(user_question)


if __name__ == "__main__":
    main()
