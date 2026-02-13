# Advanced Retrieval Features

## 🚀 Multi-Query + Reranking

Your RAG system now supports advanced retrieval strategies that improve answer quality!

## 📊 Features

### 1. **Multi-Query Retrieval**
Generates multiple variations of your question to capture different perspectives.

**Benefits:**
- ✅ Better document coverage
- ✅ Finds relevant docs that single query might miss
- ✅ Handles ambiguous questions better

**Trade-offs:**
- ⏱️ Slower (generates query variations)
- 💰 More LLM calls

### 2. **Document Reranking**
Re-scores retrieved documents by relevance to improve quality.

**Benefits:**
- ✅ Higher quality results
- ✅ Most relevant documents ranked first
- ✅ Filters out less relevant content

**Trade-offs:**
- ⏱️ Slower (scores each document)
- 💰 More LLM calls

## 🎯 Usage

### **Enable in Configuration**

```python
from src.rag_system import RAGSystem, RAGConfig

config = RAGConfig(
    use_multi_query=True,   # ✅ Generate query variations
    use_reranking=True,     # ✅ Rerank by relevance
    num_queries=3,          # Number of query variations
    retrieval_k=10,         # Retrieve more docs, return best 3
    top_k=3                 # Final number of docs to use
)

rag = RAGSystem(config)
result = rag.query("What is the most expensive transaction?")
```

### **Enable in Web UI**

Just check the boxes:
- ☑️ Multi-Query Retrieval
- ☑️ Rerank Results

### **Compare Strategies**

Run the comparison example:
```bash
python example_advanced_retrieval.py
```

## 📈 Performance Comparison

| Strategy | Speed | Quality | LLM Calls | Best For |
|----------|-------|---------|-----------|----------|
| **Basic** | ⚡⚡⚡ | ⭐⭐ | 1 | Simple Q&A, speed priority |
| **Multi-Query** | ⚡⚡ | ⭐⭐⭐ | 4+ | Better coverage |
| **Multi+Rerank** | ⚡ | ⭐⭐⭐⭐ | 10+ | Best quality, accuracy |

## 🔧 Configuration Options

```python
config = RAGConfig(
    # Enable features
    use_multi_query=False,   # Enable multi-query
    use_reranking=False,     # Enable reranking
    
    # Multi-query settings
    num_queries=3,           # Number of query variations (2-5)
    
    # Reranking settings
    retrieval_k=10,          # Docs to retrieve before reranking
    top_k=3,                 # Final docs after reranking
)
```

## 💡 Best Practices

### **When to Use Multi-Query**
- ✅ Complex or ambiguous questions
- ✅ When you need broad coverage
- ✅ Questions with multiple aspects
- ❌ Simple factual lookups
- ❌ When speed is critical

### **When to Use Reranking**
- ✅ Large document collections
- ✅ When quality matters more than speed
- ✅ Noisy or varied content
- ❌ Small, high-quality doc sets
- ❌ Real-time applications

### **When to Use Both**
- ✅ Critical queries needing best answers
- ✅ Research or analysis tasks
- ✅ Production systems with quality SLAs
- ❌ High-volume, low-latency needs
- ❌ Limited compute resources

## 🎮 Examples

### Example 1: Basic vs Advanced

```python
# Basic (fast)
config = RAGConfig(
    use_multi_query=False,
    use_reranking=False,
    top_k=3
)

# Advanced (high quality)
config = RAGConfig(
    use_multi_query=True,
    use_reranking=True,
    num_queries=3,
    retrieval_k=10,
    top_k=3
)
```

### Example 2: Multi-Query Only

```python
# Good balance of speed and quality
config = RAGConfig(
    use_multi_query=True,   # Better coverage
    use_reranking=False,    # Still fast
    num_queries=3,
    top_k=3
)
```

### Example 3: Reranking Only

```python
# Better quality without query generation
config = RAGConfig(
    use_multi_query=False,
    use_reranking=True,     # Better ranking
    retrieval_k=10,         # Retrieve more
    top_k=3                 # Return best 3
)
```

## 🔍 How It Works

### Multi-Query Pipeline
```
User question
    ↓
Generate 3 query variations (LLM)
    ↓
Retrieve docs for each query
    ↓
Deduplicate results
    ↓
Return combined results
```

### Reranking Pipeline
```
Retrieve N documents (e.g., 10)
    ↓
Score each doc for relevance (LLM)
    ↓
Sort by score
    ↓
Return top K (e.g., 3)
```

### Combined Pipeline
```
User question
    ↓
Multi-query (3 variations)
    ↓
Retrieve 10 docs per query
    ↓
Deduplicate (e.g., 25 unique docs)
    ↓
Rerank all 25 docs
    ↓
Return top 3
```

## 🐛 Troubleshooting

### Slow Performance
- Reduce `num_queries` (3 → 2)
- Reduce `retrieval_k` (10 → 5)
- Disable one feature
- Use local LLM (faster for scoring)

### High API Costs
- Use with local LLM (`use_local_llm=True`)
- Reduce number of queries
- Cache results
- Only enable for important queries

### Not Improving Results
- Check if basic retrieval already works well
- Try different combinations
- Adjust `retrieval_k` and `top_k`
- Verify document quality

## 📚 References

- **Multi-Query**: [LangChain MultiQueryRetriever](https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever)
- **Reranking**: [Contextual Compression](https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression/)

---

**The features are disabled by default, so your existing setup continues to work!** ✅
