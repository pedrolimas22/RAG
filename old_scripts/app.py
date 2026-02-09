"""
Gradio Web UI for RAG System
Run this to get a nice web interface for your RAG system!
"""

import gradio as gr
from dotenv import load_dotenv
from rag_improved import SimpleRAG, RAGConfig
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize RAG system globally
config = RAGConfig(
    chunk_size=500,
    chunk_overlap=50,
    embedding_model="text-embedding-3-small",
    llm_model="gpt-3.5-turbo",
    temperature=0,
    top_k=3,
    file_type="txt"
)

rag = SimpleRAG(documents_path="./documents", config=config)

# Try to initialize (will fail gracefully if API key missing)
try:
    rag.initialize(force_rebuild=False)
    initialized = True
except Exception as e:
    logger.error(f"Failed to initialize RAG: {e}")
    initialized = False


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


def rebuild_index(file_type: str) -> str:
    """Rebuild the vector store index."""
    try:
        logger.info(f"Rebuilding index for {file_type} files...")
        rag.config.file_type = file_type
        rag.initialize(force_rebuild=True)
        return f"✅ Index rebuilt successfully for {file_type} files!"
    except Exception as e:
        logger.error(f"Error rebuilding index: {e}")
        return f"❌ Error rebuilding index: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="RAG Question Answering System", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🤖 RAG Question Answering System
        
        Ask questions about your documents and get AI-powered answers with sources!
        """
    )
    
    with gr.Tab("💬 Ask Questions"):
        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="What is machine learning?",
                    lines=3
                )
                num_sources_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Number of source documents to retrieve"
                )
                ask_btn = gr.Button("Ask", variant="primary", size="lg")
            
        with gr.Row():
            with gr.Column():
                answer_output = gr.Textbox(
                    label="Answer",
                    lines=8,
                    show_copy_button=True
                )
            with gr.Column():
                sources_output = gr.Markdown(
                    label="Sources"
                )
        
        # Example questions
        gr.Examples(
            examples=[
                ["What is the main topic of these documents?", 3],
                ["Can you summarize the key points?", 3],
                ["What are the applications mentioned?", 5],
            ],
            inputs=[question_input, num_sources_slider],
        )
        
        ask_btn.click(
            fn=ask_question,
            inputs=[question_input, num_sources_slider],
            outputs=[answer_output, sources_output]
        )
    
    with gr.Tab("⚙️ Settings"):
        gr.Markdown("### Rebuild Vector Store Index")
        file_type_radio = gr.Radio(
            choices=["txt", "csv"],
            value="txt",
            label="File Type"
        )
        rebuild_btn = gr.Button("Rebuild Index", variant="secondary")
        rebuild_output = gr.Textbox(label="Status")
        
        rebuild_btn.click(
            fn=rebuild_index,
            inputs=[file_type_radio],
            outputs=[rebuild_output]
        )
    
    with gr.Tab("ℹ️ About"):
        gr.Markdown(
            """
            ### How it works:
            
            1. **Document Loading**: Your documents are loaded from the `documents/` folder
            2. **Chunking**: Documents are split into smaller chunks for better retrieval
            3. **Embedding**: Each chunk is converted to a vector using OpenAI's embedding model
            4. **Storage**: Vectors are stored in ChromaDB for fast similarity search
            5. **Retrieval**: When you ask a question, the most relevant chunks are found
            6. **Generation**: GPT generates an answer based on the retrieved context
            
            ### Configuration:
            - **Embedding Model**: text-embedding-3-small
            - **LLM Model**: gpt-3.5-turbo
            - **Chunk Size**: 500 characters
            - **Chunk Overlap**: 50 characters
            
            ### Requirements:
            - OpenAI API key in `.env` file
            - Documents in `documents/` folder (txt or csv)
            """
        )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
