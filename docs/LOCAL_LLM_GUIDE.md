# Using Local LLM on Apple Silicon (M1/M2/M3)

Run your RAG system **100% FREE** with local inference on your Mac! No API costs, complete privacy.

## 🚀 Quick Setup

### 1. Install Ollama

```bash
# Using Homebrew (recommended)
brew install ollama

# Or download from https://ollama.ai
```

### 2. Start Ollama Server

```bash
ollama serve
```

Keep this running in a terminal window.

### 3. Pull a Model

In another terminal:

```bash
# Recommended: Fast and high-quality (3B parameters)
ollama pull llama3.2:3b

# Or try other models:
ollama pull llama3.2:7b    # Larger, higher quality
ollama pull phi3:mini      # Microsoft Phi-3
ollama pull gemma2:2b      # Google Gemma
```

### 4. Configure RAG to Use Local LLM

**Option A: In Code**

```python
from src.rag_system import RAGSystem, RAGConfig

config = RAGConfig(
    use_local_llm=True,              # Enable local LLM
    local_llm_model="llama3.2:3b",   # Which model to use
    embedding_model="huggingface",   # Free embeddings
)

rag = RAGSystem(config)

# Now use it normally - 100% FREE!
rag.ingest("documents/myfile.pdf")
result = rag.query("What is this document about?")
print(result['result'])
```

**Option B: In Gradio Web UI**

Edit `web/app.py`:

```python
config = RAGConfig(
    use_local_llm=True,              # Change to True
    local_llm_model="llama3.2:3b",
    # ... rest of config
)
```

Then run:
```bash
python web/app.py
```

## 📊 Performance on Apple Silicon

| Model | Size | Speed (M1) | Quality | Memory |
|-------|------|------------|---------|--------|
| llama3.2:3b | 3B | ⚡⚡⚡ Fast | ⭐⭐⭐ Good | 4 GB |
| llama3.2:7b | 7B | ⚡⚡ Medium | ⭐⭐⭐⭐ Great | 8 GB |
| phi3:mini | 3.8B | ⚡⚡⚡ Fast | ⭐⭐⭐ Good | 4 GB |
| gemma2:2b | 2B | ⚡⚡⚡⚡ Very Fast | ⭐⭐ OK | 2 GB |

**Recommended:** `llama3.2:3b` - Best balance of speed and quality

## 💰 Cost Comparison

| Setup | Cost per 1000 queries |
|-------|----------------------|
| OpenAI (gpt-4o-mini) | ~$0.18 |
| **Local LLM (Ollama)** | **$0.00** ✅ |

## 🎯 Example Usage

### Run the Example Script

```bash
python example_local_llm.py
```

### From Scripts

```bash
# Make sure Ollama is running first!

# Ingest
python scripts/ingest.py documents/file.pdf

# Query (will use config from RAGConfig)
python scripts/query.py "What is the main topic?" --verbose
```

### Interactive Mode

```bash
python scripts/interactive.py
```

## 🔧 Advanced Configuration

### List Available Models

```python
from src.rag_system import LocalLLM

models = LocalLLM.list_available_models()
print("Installed models:", models)
```

### Get Recommended Models

```python
from src.rag_system import LocalLLM

small = LocalLLM.get_recommended_model("small")    # llama3.2:3b
medium = LocalLLM.get_recommended_model("medium")  # llama3.2:7b
```

### Custom Configuration

```python
from src.rag_system import LocalLLM

llm = LocalLLM(
    model_name="llama3.2:3b",
    temperature=0.0,        # 0 = deterministic
    num_ctx=4096,          # Context window (tokens)
)

# Use with LangChain
chain = prompt | llm.get_llm() | output_parser
```

## 🐛 Troubleshooting

### "Connection refused" Error

**Problem:** Ollama server isn't running

**Solution:**
```bash
ollama serve
```

### "Model not found" Error

**Problem:** Model not downloaded

**Solution:**
```bash
ollama pull llama3.2:3b
```

### Slow Performance

**Problem:** Model too large for your RAM

**Solution:** Try a smaller model:
```bash
ollama pull gemma2:2b  # Only 2 GB
```

### Check Ollama Status

```bash
# List installed models
ollama list

# Test a model
ollama run llama3.2:3b "Hello!"
```

## 🌟 Benefits of Local LLM

✅ **Zero Cost** - No API fees  
✅ **Privacy** - Data never leaves your Mac  
✅ **Offline** - Works without internet  
✅ **Fast** - Metal GPU acceleration  
✅ **Unlimited** - No rate limits or quotas  

## 📚 Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Available Models](https://ollama.ai/library)
- [LangChain Ollama Integration](https://python.langchain.com/docs/integrations/llms/ollama)

---

**Ready to go?** Run `python example_local_llm.py` to get started! 🚀
