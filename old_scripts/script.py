import os                                                                                                                                                                                                          
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.document_loaders.csv_loader import CSVLoader as cvLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma


# Load .env file from the same directory as this script
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Debug: Print what was loaded
# print(f"OPENAI_API_KEY loaded: {os.getenv('OPENAI_API_KEY')[:20] if os.getenv('OPENAI_API_KEY') else 'None'}...")
# print(f"CHROMA_API_KEY loaded: {os.getenv('CHROMA_API_KEY')[:20] if os.getenv('CHROMA_API_KEY') else 'None'}...")
# print(f"CHROMA_TENANT loaded: {os.getenv('CHROMA_TENANT')[:20] if os.getenv('CHROMA_TENANT') else 'None'}...")

# Environment variables are already loaded by load_dotenv()
# No need to reassign them to os.environ - they're already there
# Only set if you need to override or set defaults
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file")

if not os.getenv("CHROMA_API_KEY"):
    raise ValueError("CHROMA_API_KEY not found in .env file")

if not os.getenv("CHROMA_TENANT"):
    raise ValueError("CHROMA_TENANT not found in .env file")


# Version Management
VERSION = "v2"
# NOTE: We re-ingest to keep versions clean and independent,
#       but technically v2 could use v1's data since structure is identical

# Vector Database collection name
COLLECTION_NAME = f"transaction_docs_{VERSION}"


# Embeddings Model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

#--------------------------------------------------------------------------------
# v2 - Classification Model
#--------------------------------------------------------------------------------
# LLM

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

# Vector Database
vectorstore = Chroma(
  embedding_function=embeddings,
  collection_name=COLLECTION_NAME,  # Version-based naming
  chroma_cloud_api_key=os.getenv("CHROMA_API_KEY"),
  tenant=os.getenv("CHROMA_TENANT"),
  database="code_for_all_rag"
)

#--------------------------------------------------------------------------------
# v2 - Imports
#--------------------------------------------------------------------------------

import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def ingest_documents(document_path: str) -> None:
    """
    INGESTION PIPELINE - VERSION v2
    """

    print("-" * 80)
    print(f"STARTING INGESTION PIPELINE - VERSION {VERSION}")
    print("-" * 80)

    #--------------------------------------------------------------------------------
    # STEP 1: LOAD DOCUMENTS
    #--------------------------------------------------------------------------------
    print("\n[1/3] Loading file from URL...")

    loader = cvLoader(document_path)
    data = list(loader.lazy_load())
    #print(data)  # Print first 2 documents for inspection

    print(f"✓ Loaded {len(data)} pages from file")

    #--------------------------------------------------------------------------------
    # STEP 2: TOPIC DETECTION
    #--------------------------------------------------------------------------------
    topic_detection_template = ChatPromptTemplate.from_template(
      """
      Analyze the following document content and determine its primary topic.

      Document content:
      {content}

      Based on this content, what is the primary topic? Answer with a single word or short phrase (e.g., 'bitcoin', 'ethereum', 'blockchain').

      Examples:
      If the document is about Bitcoin, answer: bitcoin
      If the document is about Ethereum, answer: ethereum
      If the document is about general blockchain technology, answer: blockchain

      Primary topic:
      """
    )

    topic_detection_chain = topic_detection_template | llm | StrOutputParser()

    # sample content will be the first 3 rows from CSV
    sample_content = ""
    doc_list = list(data[:3])  # Convert generator to list to get first 3 rows
    print(doc_list)

    for doc in doc_list:
        # CSVLoader creates documents where page_content contains the row data
        sample_content += doc.page_content + " "

    # Limit to 4000 chars
    sample_content = sample_content[:4000]

    print("\nDetecting topic based on sample content...")
    print(sample_content)

    detected_topic = topic_detection_chain.invoke({
        "content": sample_content
    }).strip().lower()


    #--------------------------------------------------------------------------------
    # STEP 3: PREPROCESSING // CLEANING
    #--------------------------------------------------------------------------------

    for cell in data:
      # Remove multiple whitespaces -> \s+ matches one or more whitespace characters (spaces, tabs, newlines) and replaces them with a single space " "
      cell.page_content = re.sub(r'\s+', ' ', cell.page_content)

      # Remove standalone page numbers -> remove a trailing number only if it's the final token
      cell.page_content = re.sub(r'(?<=\.)\s*\d+\s*$', '', cell.page_content)

      # Removes whitespace from both the beginning and the end of the string
      cell.page_content = cell.page_content.strip()

    #--------------------------------------------------------------------------------
    # STEP 4: CHUNK DOCUMENTS
    #--------------------------------------------------------------------------------
    print(f"\n[2/3] Chunking documents...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    print(f"✓ Split into {len(chunks)} chunks")

    #--------------------------------------------------------------------------------
    # STEP 5: ADD METADATA
    #--------------------------------------------------------------------------------

    for chunk in chunks:
      chunk.metadata.update({
       "topic": detected_topic,
       "access_level": access_level   
      })

    #--------------------------------------------------------------------------------
    # STEP 5: CREATE EMBEDDINGS AND STORE IN CHROMA
    #--------------------------------------------------------------------------------

    print(f"\n[3/3] Creating embeddings and storing in Chroma...")
    print(f"  Collection name: {COLLECTION_NAME}")

    vectorstore.add_documents(
        documents=chunks,
        #ids=[str(i) for i in range(len(chunks))]
    )

    print(f"✓ Embeddings created and stored")

if __name__ == "__main__":
   ingest_documents("documents/Transactions.csv")