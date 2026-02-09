# RAG System

A modular, production-ready Retrieval-Augmented Generation (RAG) system built with LangChain, OpenAI, and ChromaDB.

## 🌟 Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for configuration, embeddings, vector stores, document loading, ingestion, and querying
- **Flexible Embeddings**: Support for both OpenAI and HuggingFace embeddings
- **Cloud or Local Storage**: Use ChromaDB Cloud or local persistence
- **Multiple Interfaces**: CLI scripts, interactive REPL, and Gradio web UI
- **Production Ready**: Proper logging, error handling, and configuration management
- **Type Hints**: Full type annotations for better code quality
- **Easy Installation**: Standard Python packaging with `pip`

## 📁 Project Structure

```
RAG/
├── src/
│   └── rag_system/          # Main package
│       ├── __init__.py      # Package initialization
│       ├── config.py        # Configuration management
│       ├── embeddings.py    # Embedding models
│       ├── vectorstore.py   # Vector store management
│       ├── document_loader.py # Document loading
│       ├── ingestion.py     # Ingestion pipeline
│       ├── query.py         # Query/inference pipeline
│       └── rag.py           # Main RAG system class
├── scripts/
│   ├── ingest.py           # CLI for document ingestion
│   ├── query.py            # CLI for querying
│   └── interactive.py      # Interactive REPL
├── web/
│   └── app.py              # Gradio web interface
├── documents/              # Documents go here
├── chroma_db/              # Vector database storage
├── tests/                  # Unit tests
├── .env                    # Environment variables (create from .env.example)
├── .env.example            # Template for environment variables
├── .gitignore              # Git ignore rules
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project metadata and build config
├── setup.py                # Setup script
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd RAG

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install the package in development mode
pip install -e .
```

### 2. Configuration

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
CHROMA_API_KEY=your_chroma_api_key_here  # Optional: only for ChromaDB Cloud
CHROMA_TENANT=your_chroma_tenant_here    # Optional: only for ChromaDB Cloud
```

### 3. Ingest Documents

```bash
# Using CLI script
python scripts/ingest.py documents/Transactions.csv

# With options
python scripts/ingest.py documents/Transactions.csv --clear --chunk-size 1000 --embedding huggingface
```

### 4. Query the System

**Command Line:**
```bash
# Single query
python scripts/query.py "What is the most expensive transaction?" --verbose --show-sources

# Interactive mode
python scripts/interactive.py
```

**Web UI:**
```bash
python web/app.py
# Open http://localhost:7860 in your browser
```

**Python API:**
```python
from src.rag_system import RAGSystem, RAGConfig

# Initialize
config = RAGConfig(
    chunk_size=1000,
    embedding_model="huggingface",
    version="v3"
)
rag = RAGSystem(config)

# Ingest documents
rag.ingest("documents/Transactions.csv")

# Query
result = rag.query("What is the most expensive transaction?")
print(result['result'])

# Show sources
for doc in result['source_documents']:
    print(doc.page_content)
```

## 🔧 Configuration Options

The `RAGConfig` class supports extensive configuration:

```python
config = RAGConfig(
    # Document processing
    chunk_size=1000,              # Size of text chunks
    chunk_overlap=200,            # Overlap between chunks
    file_type="csv",              # File type: "csv" or "txt"
    
    # Embeddings
    embedding_model="huggingface", # "openai" or "huggingface"
    openai_embedding_model="text-embedding-3-small",
    huggingface_model="sentence-transformers/all-MiniLM-L6-v2",
    
    # LLM
    llm_model="gpt-4o-mini",      # OpenAI model to use
    temperature=0.0,              # Lower = more deterministic
    
    # Retrieval
    top_k=3,                      # Number of documents to retrieve
    
    # Vector store
    collection_name="rag_documents",
    version="v3",                 # Version tag for collections
    use_chroma_cloud=True,        # Use cloud or local
    database_name="my_rag_db",
    
    # Paths
    documents_path="./documents",
    persist_directory="./chroma_db"
)
```

## 📚 Usage Examples

### CLI Scripts

```bash
# Ingest with different settings
python scripts/ingest.py documents/data.csv \
    --chunk-size 500 \
    --chunk-overlap 50 \
    --embedding openai \
    --version v4 \
    --clear

# Query with options
python scripts/query.py "Your question here" \
    --top-k 5 \
    --verbose \
    --show-sources \
    --version v4

# Interactive session
python scripts/interactive.py
```

### Python API

```python
from src.rag_system import RAGSystem, RAGConfig

# Create custom configuration
config = RAGConfig(
    embedding_model="openai",
    llm_model="gpt-4",
    top_k=5
)

# Initialize system
rag = RAGSystem(config)

# Ingest multiple files
rag.ingest("documents/file1.csv", clear_existing=True)
rag.ingest("documents/file2.csv", clear_existing=False)

# Query with verbose logging
result = rag.query("Your question?", verbose=True)
print(result['result'])

# Search without LLM generation
docs = rag.search("search query", k=10)
for doc in docs:
    print(doc.page_content)

# Update configuration on the fly
rag.update_config(top_k=10, temperature=0.7)
```

## 🧪 Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# With coverage
pytest --cov=src/rag_system
```

### Code Formatting

```bash
# Format code
black src/ scripts/ web/

# Sort imports
isort src/ scripts/ web/

# Lint
flake8 src/ scripts/ web/

# Type checking
mypy src/
```

## 🏗️ Architecture

### Core Components

1. **Configuration** (`config.py`): Centralized configuration management with environment variable loading
2. **Embeddings** (`embeddings.py`): Factory for creating embedding models
3. **Vector Store** (`vectorstore.py`): ChromaDB integration and management
4. **Document Loader** (`document_loader.py`): Load and chunk documents
5. **Ingestion Pipeline** (`ingestion.py`): Complete document processing workflow
6. **Query Pipeline** (`query.py`): Retrieval and answer generation
7. **RAG System** (`rag.py`): Main interface combining all components

### Data Flow

```
Documents → DocumentLoader → Chunks → VectorStore (via Embeddings)
                                            ↓
User Query → VectorStore (similarity search) → Relevant Chunks
                                            ↓
                            Chunks + Query → LLM → Answer
```

## 🤝 Contributing

This is a modular system designed for easy extension:

- Add new embedding models in `embeddings.py`
- Support new document types in `document_loader.py`
- Customize prompts in `query.py`
- Add new retrieval strategies in `vectorstore.py`

## 📝 License

MIT License - feel free to use this project as you wish!

## 🙏 Acknowledgments

Built with:
- [LangChain](https://github.com/langchain-ai/langchain) - LLM framework
- [OpenAI](https://openai.com/) - LLM and embeddings
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Gradio](https://gradio.app/) - Web UI
- [HuggingFace](https://huggingface.co/) - Open source embeddings

## 📧 Support

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Happy RAG-ing! 🚀**
