# LLM-Powered Query System with Tools

This enhanced query system uses LangChain agents with tools to provide intelligent answers to questions about your book content, rather than just returning similarity search results.

## 🚀 Features

- **LLM Agent**: Uses OpenAI's GPT models to understand and answer questions
- **Smart Tools**: Two specialized tools for searching and summarizing book content
- **Intelligent Answers**: Combines similarity search with LLM reasoning
- **Fallback Support**: Falls back to basic search if LLM is unavailable

## 🛠️ Tools Available

### 1. ChromaDBSearchTool (`chroma_search`)
- **Purpose**: Search for specific content, chapters, or topics
- **Use Case**: When you need to find particular information in the book
- **Returns**: Formatted search results with page numbers and relevance scores

### 2. BookSummaryTool (`book_summary`)
- **Purpose**: Get overview and summary information
- **Use Case**: When you want to understand the book's structure or main topics
- **Returns**: Summary of relevant book sections

## 📋 Prerequisites

1. **OpenAI API Key**: You need an OpenAI API key to use the LLM features
2. **Dependencies**: Install the required packages (see requirements.txt)
3. **Book Embeddings**: Run the main pipeline first to create ChromaDB embeddings

## 🔑 Setup

### 1. Set OpenAI API Key

**Windows:**
```powershell
set OPENAI_API_KEY=your-api-key-here
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY=your-api-key-here
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Create Book Embeddings
```bash
python main.py
```

## 🎯 Usage

### Option 1: Interactive LLM Query Interface
```bash
python utils/query.py
```

### Option 2: Example Script with Pre-defined Questions
```bash
python example_llm_query.py
```

### Option 3: Programmatic Usage
```python
from utils.query import load_collection, create_llm_agent, ask_question_with_llm

# Load collection
collection = load_collection()

# Create LLM agent
agent = create_llm_agent(collection)

# Ask questions
answer = ask_question_with_llm(agent, "What is this book about?")
print(answer)
```

## 🔍 How It Works

1. **Question Input**: User asks a question about the book
2. **Tool Selection**: LLM agent decides which tools to use
3. **Content Retrieval**: Tools search ChromaDB for relevant content
4. **LLM Processing**: LLM analyzes the retrieved content and formulates an answer
5. **Response**: Intelligent, contextual answer based on book content

## 💡 Example Questions

- "What is this book about?"
- "What are the main topics covered?"
- "Can you explain the key concepts?"
- "What examples are provided?"
- "How does the book structure its content?"

## ⚙️ Configuration

### Model Settings
```python
agent = create_llm_agent(
    collection,
    model_name="gpt-3.5-turbo",  # or "gpt-4" for better reasoning
    temperature=0.1               # Lower = more focused, Higher = more creative
)
```

### Search Parameters
```python
# In the tools, you can adjust:
n_results=5      # Number of search results to retrieve
chunk_size=500   # Size of text chunks to return
```

## 🚨 Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not found"**
   - Set your API key in environment variables
   - Check that the key is valid and has credits

2. **"Failed to create LLM agent"**
   - Verify your OpenAI API key
   - Check internet connection
   - Ensure you have sufficient API credits

3. **"No collection found"**
   - Run `python main.py` first to create embeddings
   - Check that `chroma_db/` directory exists

4. **Poor answer quality**
   - Try adjusting the temperature setting
   - Use more specific questions
   - Ensure your book content is well-chunked

### Fallback Behavior

If the LLM agent fails to initialize, the system automatically falls back to basic similarity search, ensuring you can still query your book content.

## 🔧 Advanced Usage

### Custom Tools
You can create additional tools by extending the `BaseTool` class:

```python
class CustomTool(BaseTool):
    name = "custom_tool"
    description = "Description of what this tool does"
    
    def _run(self, query: str) -> str:
        # Your custom logic here
        return "Custom result"
```

### Multiple Collections
The system can work with multiple book collections by loading them separately:

```python
collection1 = load_collection(collection_name="book1")
collection2 = load_collection(collection_name="book2")
```

## 📊 Performance Tips

- **Batch Processing**: Process multiple questions at once for efficiency
- **Caching**: The system caches embeddings for faster subsequent queries
- **Model Selection**: Use GPT-3.5-turbo for speed, GPT-4 for quality
- **Temperature**: Lower temperature (0.1) for factual answers, higher (0.7) for creative responses

## 🤝 Contributing

Feel free to extend the system with:
- Additional tools for specific use cases
- Support for other LLM providers
- Enhanced error handling and logging
- Integration with other vector databases
