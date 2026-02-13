#!/usr/bin/env python
"""
Gradio Web UI for RAG System.
Run this to get a nice web interface for your RAG system!
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables BEFORE importing RAGConfig
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
    print(f"✓ Loaded .env from: {env_path}")
else:
    print(f"⚠ No .env file found at {env_path}")

import gradio as gr
import logging
from src.rag_system import RAGSystem, RAGConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize RAG system globally
# Using ultra-low-cost settings to minimize API usage
# Set use_local_llm=True for FREE inference on M1/M2/M3 chips (requires Ollama)
try:
    config = RAGConfig(
        chunk_size=600,         # Smaller chunks = less context
        chunk_overlap=100,      # Less overlap = fewer chunks
        embedding_model="huggingface",  # FREE embeddings (no OpenAI cost!)
        use_local_llm=True,    # Set True for FREE local LLM (requires Ollama)
        llm_model="gpt-4o-mini",        # OpenAI model (if use_local_llm=False)
        local_llm_model="llama3.2:3b",  # Ollama model (if use_local_llm=True)
        temperature=0,
        top_k=2                 # Only 2 source docs = less tokens
    )
    rag = RAGSystem(config)
    initialized = True
    logger.info("✅ RAG system initialized successfully")
    logger.info(f"✅ OpenAI API Key loaded: {config.openai_api_key[:15]}..." if config.openai_api_key else "❌ No OpenAI API key")
except Exception as e:
    logger.error(f"❌ Failed to initialize RAG: {e}")
    logger.error("Please check that .env file exists with OPENAI_API_KEY set")
    initialized = False
    rag = None


def ask_question(
    question: str, 
    num_sources: int = 3,
    use_multi_query: bool = False,
    use_reranking: bool = False
) -> tuple[str, str]:
    """
    Process a question and return the answer and sources.
    
    Args:
        question: The user's question
        num_sources: Number of source documents to retrieve
        use_multi_query: Enable multi-query retrieval
        use_reranking: Enable document reranking
        
    Returns:
        Tuple of (answer, sources_info)
    """
    if not initialized:
        return "❌ RAG system not initialized. Please check your API keys in .env file.", ""
    
    if not question.strip():
        return "Please enter a question.", ""
    
    try:
        # Update config for this query
        rag.config.top_k = num_sources
        rag.config.use_multi_query = use_multi_query
        rag.config.use_reranking = use_reranking
        
        # Reinitialize query pipeline if advanced features changed
        if use_multi_query or use_reranking:
            rag.query_pipeline = None  # Force reinitialization
            from src.rag_system.query import QueryPipeline
            rag.query_pipeline = QueryPipeline(rag.config)
        
        # Get response
        response = rag.query(question, verbose=False)
        
        answer = response['result']
        
        # Format sources
        sources_info = "### 📚 Source Documents:\n\n"
        for i, doc in enumerate(response['source_documents'], 1):
            source = doc.metadata.get('source', 'Unknown')
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            sources_info += f"**Source {i}:** `{source}`\n\n{content_preview}\n\n---\n\n"
        
        return answer, sources_info
    
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return f"❌ Error: {str(e)}", ""


def ingest_file(uploaded_file, file_path: str, clear_existing: bool) -> str:
    """
    Ingest a document file from upload or path.
    
    Args:
        uploaded_file: File uploaded via Gradio File component (filepath string)
        file_path: Path to file (alternative to upload)
        clear_existing: Whether to clear existing documents
        
    Returns:
        Status message
    """
    if not initialized:
        return "❌ RAG system not initialized. Please check your API keys."
    
    # Determine which input to use
    target_file = None
    source_type = None
    
    if uploaded_file is not None and uploaded_file:
        # Gradio File component with type="filepath" returns a string path
        target_file = str(uploaded_file)
        source_type = "upload"
        logger.info(f"Using uploaded file: {target_file}")
        logger.info(f"File type: {type(uploaded_file)}, Value: {uploaded_file}")
    elif file_path and file_path.strip():
        # Use file path - resolve relative to project root
        project_root = Path(__file__).parent.parent
        target_path = Path(file_path)
        source_type = "path"
        
        # Try as absolute path first
        if target_path.is_absolute() and target_path.exists():
            target_file = str(target_path)
        # Try relative to project root
        elif (project_root / target_path).exists():
            target_file = str(project_root / target_path)
        # Try relative to current directory
        elif target_path.exists():
            target_file = str(target_path)
        else:
            return f"❌ File not found: {file_path}\n\nTried:\n- {target_path}\n- {project_root / target_path}"
    else:
        return "❌ Please upload a file or enter a file path."
    
    # Verify file exists and check extension
    target_path_obj = Path(target_file)
    if not target_path_obj.exists():
        return f"❌ File not found: {target_file}"
    
    # Check file extension
    extension = target_path_obj.suffix.lower()
    supported_extensions = ['.pdf', '.csv', '.txt', '.md']
    if extension not in supported_extensions:
        return f"❌ Unsupported file type: {extension}\n\nSupported: {', '.join(supported_extensions)}"
    
    try:
        logger.info(f"Ingesting {extension} file from {source_type}: {target_file}")
        logger.info(f"File size: {target_path_obj.stat().st_size / 1024:.2f} KB")
        
        num_chunks = rag.ingest(target_file, clear_existing=clear_existing)
        file_name = target_path_obj.name
        
        return f"✅ Successfully ingested **{num_chunks} chunks** from `{file_name}`!\n\n📁 File: `{target_file}`\n📄 Type: {extension}\n📊 Source: {source_type}"
    except Exception as e:
        logger.error(f"Error ingesting file: {e}", exc_info=True)
        import traceback
        error_details = traceback.format_exc()
        return f"❌ Error ingesting {extension} file:\n\n```\n{str(e)}\n\nDetails:\n{error_details}\n```"


# Create Gradio interface
with gr.Blocks(title="RAG Question Answering System", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🤖 RAG Question Answering System
        
        Ask questions about your documents and get AI-powered answers with sources!
        """
    )
    
    with gr.Tab("💬 Ask Questions"):
        gr.Markdown("### Ask a question about your documents")
        
        with gr.Row():
            with gr.Column(scale=4):
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="What would you like to know?",
                    lines=2
                )
            with gr.Column(scale=1):
                num_sources = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Number of Sources"
                )
        
        # Advanced retrieval options
        with gr.Row():
            gr.Markdown("### 🚀 Advanced Retrieval (Beta)")
        
        with gr.Row():
            multi_query_checkbox = gr.Checkbox(
                label="🔍 Multi-Query Retrieval",
                value=False,
                info="Generate multiple query variations for better coverage (slower)"
            )
            reranking_checkbox = gr.Checkbox(
                label="📊 Rerank Results",
                value=False,
                info="Re-score documents by relevance (slower but higher quality)"
            )
        
        gr.Markdown("*Note: Advanced features increase processing time but improve answer quality*")
        
        ask_btn = gr.Button("Ask", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column():
                answer_output = gr.Markdown(label="Answer")
            with gr.Column():
                sources_output = gr.Markdown(label="Sources")
        
        ask_btn.click(
            fn=ask_question,
            inputs=[question_input, num_sources, multi_query_checkbox, reranking_checkbox],
            outputs=[answer_output, sources_output]
        )
        
        # Example questions
        gr.Examples(
            examples=[
                ["What is the most expensive transaction?"],
                ["Show me all outflow transactions."],
                ["What categories are present in the data?"],
            ],
            inputs=question_input,
            label="Example Questions"
        )
    
    with gr.Tab("📤 Ingest Documents"):
        gr.Markdown("### Add documents to the RAG system")
        gr.Markdown("Upload a file or enter a file path to ingest documents.")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Option 1: Upload File**")
                file_upload = gr.File(
                    label="Upload Document",
                    file_types=[".txt", ".csv", ".pdf", ".md"],
                    type="filepath"
                )
                gr.Markdown("*Drag & drop or click to browse*")
            
            with gr.Column():
                gr.Markdown("**Option 2: Enter File Path**")
                file_path_input = gr.Textbox(
                    label="File Path (relative to project root)",
                    placeholder="e.g., documents/bitcoin.pdf",
                    lines=1
                )
                gr.Markdown("*Use this for files already in the project*")
        
        clear_checkbox = gr.Checkbox(
            label="Clear existing documents before ingesting",
            value=False,
            info="⚠️ This will delete all previously ingested documents"
        )
        
        ingest_btn = gr.Button("Ingest Document", variant="primary", size="lg")
        ingest_output = gr.Markdown(label="Status")
        
        ingest_btn.click(
            fn=ingest_file,
            inputs=[file_upload, file_path_input, clear_checkbox],
            outputs=ingest_output
        )
        
        # Examples
        gr.Markdown("### 💡 Supported File Types")
        gr.Markdown("""
        - **PDF** (.pdf) - Bitcoin whitepaper, research papers, etc.
        - **CSV** (.csv) - Transaction data, spreadsheets
        - **Text** (.txt) - Plain text documents
        - **Markdown** (.md) - Documentation files
        
        **Try uploading:** Drag and drop `documents/bitcoin.pdf` into the upload box above!
        """)
    
    with gr.Tab("ℹ️ System Info"):
        if initialized:
            llm_type = "Local (Ollama)" if config.use_local_llm else "OpenAI Cloud"
            llm_name = config.local_llm_model if config.use_local_llm else config.llm_model
            cost_status = "💰 FREE" if config.use_local_llm else "💳 Paid API"
            
            info_text = f"""
            ### System Configuration
            
            - **LLM Type:** {llm_type} {cost_status}
            - **LLM Model:** {llm_name}
            - **Embedding Model:** {config.embedding_model} (FREE)
            - **Version:** {config.version}
            - **Chunk Size:** {config.chunk_size}
            - **Chunk Overlap:** {config.chunk_overlap}
            - **Collection Name:** {config.collection_name}
            - **Top-K Results:** {config.top_k}
            
            ### � Advanced Retrieval Features (Beta)
            
            Enable in the "Ask Questions" tab for better results:
            
            **Multi-Query Retrieval:**
            - Generates multiple query variations for better coverage
            - Finds documents a single query might miss
            - Trade-off: Slower (~3-4x LLM calls)
            
            **Document Reranking:**
            - Re-scores documents by relevance after retrieval
            - Returns only the most relevant content
            - Trade-off: Slower (1 LLM call per document)
            
            **When to Use:**
            - ✅ Complex questions needing comprehensive answers
            - ✅ When quality matters more than speed
            - ❌ Simple factual lookups
            - ❌ Real-time/high-volume applications
            
            **Tip:** Use local LLM to reduce costs when using advanced features!
            
            ### �💡 To Use Local LLM (M1/M2/M3):
            
            1. Install Ollama: `brew install ollama`
            2. Start Ollama: `ollama serve`
            3. Pull a model: `ollama pull llama3.2:3b`
            4. Set `use_local_llm=True` in web/app.py
            
            **Benefits:** 100% FREE, private, runs on your Mac!
            """
        else:
            info_text = """
            ### ❌ System Not Initialized
            
            Please check:
            - `.env` file exists with required API keys
            - `OPENAI_API_KEY` is set
            - `CHROMA_API_KEY` is set (if using ChromaDB Cloud)
            - `CHROMA_TENANT` is set (if using ChromaDB Cloud)
            """
        
        gr.Markdown(info_text)

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
