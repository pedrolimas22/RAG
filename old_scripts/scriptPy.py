import os                                                                                                                                                                                                          
from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


# Load .env file from the same directory as this script
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

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
VERSION = "v3"
# NOTE: We re-ingest to keep versions clean and independent

# Vector Database collection name
COLLECTION_NAME = f"transaction_docs_{VERSION}"

#--------------------------------------------------------------------------------
# ChatOpenAI
#--------------------------------------------------------------------------------

# Embeddings Model
# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-small"
# )

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    verbose=False
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
# Ingestion
#--------------------------------------------------------------------------------

def ingest_documents(path: str):
    """
    INGESTION PIPELINE

    """

    print("-" * 80)
    print(f"STARTING INGESTION PIPELINE - VERSION {VERSION}")
    print("-" * 80)

    #--------------------------------------------------------------------------------
    # STEP 1: LOAD DOCUMENTS
    #--------------------------------------------------------------------------------
    print("\n[1/3] Loading file from URL...")

    loader = CSVLoader(
        file_path=path,
        csv_args={
              "delimiter": ",",
              "quotechar": '"'              
            },
        )
    documents = loader.load()

    print(f"\u2713 Loaded {len(documents)} pages from file")

    #--------------------------------------------------------------------------------
    # STEP 2: CHUNK DOCUMENTS
    #--------------------------------------------------------------------------------
    print(f"\n[2/3] Chunking documents...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    print(f"\u2713 Split into {len(chunks)} chunks")

    #--------------------------------------------------------------------------------
    # STEP 3: CREATE EMBEDDINGS AND STORE IN CHROMA
    #--------------------------------------------------------------------------------

    print(f"\n[3/3] Creating embeddings and storing in Chroma...")
    print(f"  Collection name: {COLLECTION_NAME}")

    # Split documents into batches to avoid exceeding ChromaDB's batch size limit
    batch_size = 300
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    try:
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_ids = [f"{i}_{j}" for j in range(len(batch_chunks))]
            vectorstore.add_documents(
                documents=batch_chunks,
                ids=batch_ids
            )
            batch_num = i // batch_size + 1
            print(f"    Added batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)")
        
        print(f"\u2713 Embeddings created and stored")
    except Exception as e:
        print(f"\u2717 Error during ingestion: {str(e)}")
        raise

    #--------------------------------------------------------------------------------
# v2 - added ChatPromptTemplate and StrOutputParser imports
#--------------------------------------------------------------------------------
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def inference(query: str) -> str:
    """
    INFERENCE PIPELINE - VERSION v2

    WHAT CHANGED FROM v1?
    - Return type: List[Document] → str (natural language answer)
    - Added: Context formatting
    - Added: Prompt engineering
    - Added: LLM generation

    THE RAG FLOW:
    1. Retrieve: Get relevant documents (same as v1)
    2. Format: Combine documents into context string
    3. Prompt: Create structured instruction for LLM
    4. Chain: prompt -> llm -> output parser
    5. Generate: LLM produces natural language answer
    6. Return: User gets readable answer

    Args:
        query (str): User's question

    Returns:
        str: Natural language answer (v1 returned List[Document])
    """

    print("="*80)
    print(f"RUNNING INFERENCE - VERSION {VERSION}")
    print("="*80)

    #--------------------------------------------------------------------------------
    # STEP 1: SIMILARITY SEARCH
    #--------------------------------------------------------------------------------

    print(f"\n[1/5] Performing similarity search...")
    print(f"  Query: '{query}'")

    results = vectorstore.similarity_search(query, k=3)

    print(f"✓ Found {len(results)} relevant chunks")

    #--------------------------------------------------------------------------------
    # STEP 2: FORMAT CONTEXT (NEW in v2)
    #--------------------------------------------------------------------------------

    print(f"\n[2/5] Formatting context for LLM...")

    context = "\n\n".join([doc.page_content for doc in results])

    # INSPECT: What does formatted context look like?
    print(f"\nFirst 1500 chars: {context[:1500]}...")
    print(f"✓ Context formatted ({len(context)} characters)")

    #--------------------------------------------------------------------------------
    # STEP 3: PROMPT TEMPLATE (NEW in v2)
    #--------------------------------------------------------------------------------
    # ChatPromptTemplate:
    # - Defines structure with variables in {curly braces}
    # - Variables are filled when chain is invoked
    # - Reusable across all queries

    print("\n[3/5] Creating prompt template...")
    print("  Variables: {context}, {query}")

    prompt_template = ChatPromptTemplate.from_template(
      """
      Based on the following documents, answer the question clearly and concisely (max of 2 or 3 paragraphs).
      If the answer is not in the documents, say so and don't use any other information.

      Documents: {context}

      Question: {query}

      Answer:
      """
    )

    print("✓ Prompt template created")
    print("  This template will be reused for every query")
    print("  Variables will be filled automatically by the chain")

    #--------------------------------------------------------------------------------
    # STEP 4: COMPOSE CHAIN (NEW in v2)
    #--------------------------------------------------------------------------------
    # The pipe (|) operator connects components (output from the last is the input of the next)
    # Read left to right: prompt → llm → parser

    # StrOutputParser:
    # - LLMs return AIMessage objects (complex)
    # - Parser extracts just the string content
    # - Clean string output for users

    print("\n[4/5] Composing chain...")

    chain = prompt_template | llm | StrOutputParser()

    print("\n✓ Chain composed!")
    print("\n  Chain structure:")
    print("  prompt_template  (formats variables) -> llm (generates response) -> output_parser (extracts string)")
    print("  Returns: String (natural language answer)")

    #--------------------------------------------------------------------------------
    # STEP 5: GENERATE ANSWER BY INVOKING CHAIN (NEW in v2)
    #--------------------------------------------------------------------------------
    # One line replaces multiple steps of manual prompting

    print(f"\n[5/5] Invoking RAG chain...")
    print("\n  Invoking chain with context and query...")
    print("  The chain will:")
    print("    1. Format the prompt template")
    print("    2. Send to LLM")
    print("    3. Parse response to string")
    print("    4. Return answer")

    # Pass variables as dictionary to the chain
    response = chain.invoke({
        "context": context,  # Retrieved documents
        "query": query       # User's question
    })

    print(f"\n✓ Answer generated ({len(response)} characters)")

    print("\n" + "="*80)
    print("INFERENCE COMPLETE")
    print("="*80)

    return response

if __name__ == "__main__":
   ingest_documents("documents/Transactions.csv")
   res = inference("What is the most expensive transaction?")
   res

