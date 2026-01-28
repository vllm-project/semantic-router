# OpenAI RAG Demo Application

This demo application showcases the OpenAI RAG (Retrieval-Augmented Generation) functionality integrated with the Semantic Router. It demonstrates both `direct_search` and `tool_based` workflow modes following the OpenAI Responses API cookbook.

## Features Demonstrated

1. **File Store API Operations**
   - Upload PDF files to OpenAI File Store
   - List and manage uploaded files
   - Delete files when done

2. **Vector Store API Operations**
   - Create vector stores
   - Attach files to vector stores
   - Search vector stores for relevant content

3. **RAG Integration with Semantic Router**
   - Direct search mode (synchronous retrieval)
   - Tool-based mode (asynchronous via file_search tool)
   - Context injection into LLM requests
   - Response quality comparison

## Prerequisites

```bash
# Install Python dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-openai-api-key"
export SEMANTIC_ROUTER_URL="http://localhost:8080"  # Default router URL
```

## Quick Start

### 1. Interactive Demo Script

Run the interactive demo that walks through the complete workflow:

```bash
python3 demo_rag_openai.py
```

**What it does:**
- Uploads sample PDF files to OpenAI File Store
- Creates a vector store and attaches files
- Tests RAG retrieval using direct_search mode
- Tests RAG retrieval using tool_based mode
- Compares responses with and without RAG context
- Cleans up resources

### 2. Streamlit Web App (Recommended)

Launch an interactive web interface:

```bash
streamlit run app.py
```

**Features:**
- 📄 File upload interface
- 🔍 Vector store management
- 💬 Interactive chat with RAG
- 📊 Response comparison (with/without RAG)
- 📈 Metrics and performance visualization


## Demo Scenarios

### Scenario 1: Document Q&A

Upload technical documentation and ask questions:

```python
# Upload a PDF about your product
file_id = upload_file("product_docs.pdf")

# Create vector store
vector_store_id = create_vector_store("Product Docs")

# Ask questions
response = ask_with_rag(
    "How do I configure authentication?",
    vector_store_id=vector_store_id,
    mode="direct_search"
)
```

### Scenario 2: Knowledge Base Search

Create a knowledge base from multiple documents:

```python
# Upload multiple documents
files = [
    upload_file("faq.pdf"),
    upload_file("troubleshooting.pdf"),
    upload_file("api_reference.pdf")
]

# Create vector store with all files
vector_store_id = create_vector_store("Knowledge Base", file_ids=files)

# Search across all documents
response = ask_with_rag(
    "What are common error codes?",
    vector_store_id=vector_store_id
)
```

### Scenario 3: Tool-Based Workflow

Use the Responses API workflow for LLM-controlled retrieval:

```python
# Configure tool_based mode
response = ask_with_rag(
    "Summarize the key points from the documentation",
    vector_store_id=vector_store_id,
    mode="tool_based"  # LLM decides when to search
)
```

## Configuration

The demo uses the following Semantic Router configuration:

```yaml
decisions:
  - name: rag-openai-decision
    rag:
      enabled: true
      backend: openai
      backend_config:
        vector_store_id: "vs_abc123"  # Set dynamically
        workflow_mode: "direct_search"  # or "tool_based"
        max_num_results: 5
      injection_mode: "system_prompt"  # or "tool_role"
      similarity_threshold: 0.7
      top_k: 5
      max_context_length: 2000
```

## Expected Results

### Without RAG
- Generic responses
- May hallucinate facts
- No document-specific context

### With RAG (Direct Search)
- Responses include relevant document excerpts
- Accurate citations
- Context-aware answers
- Faster response (synchronous retrieval)

### With RAG (Tool-Based)
- LLM controls when to search
- More natural conversation flow
- Results in response annotations
- Better for complex multi-step queries

## Performance Metrics

The demo tracks:
- **Retrieval Latency**: Time to search vector store
- **Context Length**: Amount of retrieved context
- **Similarity Scores**: Relevance of retrieved documents
- **Response Quality**: Comparison with baseline

## Cleanup

The demo includes cleanup functions:

```python
# Delete all uploaded files
cleanup_files(file_ids)

# Delete vector store
delete_vector_store(vector_store_id)
```

## Troubleshooting

### Issue: "Vector store not found"
- Ensure files are fully processed (check `file_counts.completed`)
- Wait a few seconds after attaching files

### Issue: "No results found"
- Check similarity threshold (may be too high)
- Verify files contain relevant content
- Try increasing `top_k` or `max_num_results`

### Issue: "API key invalid"
- Verify `OPENAI_API_KEY` is set correctly
- Check API key has access to File Store and Vector Store APIs

## Next Steps

1. **Customize for your use case**: Modify the demo to use your own documents
2. **Integrate with your app**: Use the API client code in your application
3. **Monitor performance**: Use Prometheus metrics to track RAG performance
4. **Optimize retrieval**: Tune similarity thresholds and top_k values

## References

- [OpenAI Responses API Cookbook](https://cookbook.openai.com/examples/rag_on_pdfs_using_file_search)
- [Semantic Router RAG Documentation](../../../../docs/RAG_OPENAI_GUIDE.md)
- [OpenAI File Store API](https://platform.openai.com/docs/api-reference/files)
- [OpenAI Vector Store API](https://platform.openai.com/docs/api-reference/vector-stores)
