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


def ask_question(question: str, num_sources: int = 3) -> tuple[str, str]:
    """
    Process a question and return the answer and sources.
    
    Args:
        question: The user's question
        num_sources: Number of source documents to retrieve
        
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


def ingest_file(file_path: str, clear_existing: bool) -> str:
    """Ingest a document file."""
    if not initialized:
        return "❌ RAG system not initialized. Please check your API keys."
    
    if not file_path or not Path(file_path).exists():
        return "❌ Please provide a valid file path."
    
    try:
        logger.info(f"Ingesting file: {file_path}")
        num_chunks = rag.ingest(file_path, clear_existing=clear_existing)
        return f"✅ Successfully ingested {num_chunks} chunks from {file_path}!"
    except Exception as e:
        logger.error(f"Error ingesting file: {e}")
        return f"❌ Error: {str(e)}"


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
        
        ask_btn = gr.Button("Ask", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column():
                answer_output = gr.Markdown(label="Answer")
            with gr.Column():
                sources_output = gr.Markdown(label="Sources")
        
        ask_btn.click(
            fn=ask_question,
            inputs=[question_input, num_sources],
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
        
        file_path_input = gr.Textbox(
            label="File Path",
            placeholder="e.g., documents/Transactions.csv",
            lines=1
        )
        
        clear_checkbox = gr.Checkbox(
            label="Clear existing documents before ingesting",
            value=False
        )
        
        ingest_btn = gr.Button("Ingest Document", variant="primary")
        ingest_output = gr.Markdown(label="Status")
        
        ingest_btn.click(
            fn=ingest_file,
            inputs=[file_path_input, clear_checkbox],
            outputs=ingest_output
        )
    
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
            
            ### 💡 To Use Local LLM (M1/M2/M3):
            
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
