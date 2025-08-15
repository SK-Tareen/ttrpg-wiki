import chromadb
from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv
from utils.embedding import query_chroma_collection, get_collection_info

# Load environment variables from .env file
load_dotenv()

class ChromaDBSearchTool(BaseTool):
    """Tool for searching ChromaDB collection and returning relevant chunks."""
    
    name: str = "chroma_search"
    description: str = "Search the book content for relevant information. Use this tool to find specific content, chapters, or topics from the book. This tool returns raw search results that you should use to formulate your answer."
    
    def __init__(self, collection):
        super().__init__()
        self._collection = collection
    
    def _run(self, query: str) -> str:
        """Execute the search and return formatted results."""
        try:
            # Search for relevant chunks
            results = query_chroma_collection(self._collection, query, n_results=8)
            
            if not results or not results['documents'] or not results['documents'][0]:
                return "No relevant content found for this query."
            
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0] if 'distances' in results else [0] * len(documents)
            
            # Format the results with more context
            formatted_results = []
            for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                relevance = 1 - distance if distance != 0 else "N/A"
                page = metadata.get('page', 'Unknown')
                chunk_id = metadata.get('chunk_id', f'chunk_{i}')
                
                formatted_results.append(
                    f"--- Content from Page {page} (Chunk {chunk_id}) ---\n"
                    f"{doc}\n"
                )
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            return f"Error searching collection: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """Async version of the search tool."""
        return self._run(query)

class BookSummaryTool(BaseTool):
    """Tool for getting book overview and summary information."""
    
    name: str = "book_summary"
    description: str = "Get an overview or summary of the book content. Use this tool to understand the book's structure, main topics, or general content. This tool returns raw content that you should use to formulate your answer."
    
    def __init__(self, collection):
        super().__init__()
        self._collection = collection
    
    def _run(self, query: str = "book overview introduction content") -> str:
        """Get book summary information."""
        try:
            results = query_chroma_collection(self._collection, query, n_results=8)
            
            if not results or not results['documents'] or not results['documents'][0]:
                return "Unable to retrieve book summary information."
            
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            
            # Format summary results with more context
            summary_parts = []
            for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
                page = metadata.get('page', 'Unknown')
                summary_parts.append(f"--- Content from Page {page} ---\n{doc}\n")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            return f"Error getting book summary: {str(e)}"
    
    async def _arun(self, query: str = "book overview introduction content") -> str:
        """Async version of the summary tool."""
        return self._run(query)

def load_collection(persist_dir="chroma_db", collection_name="book_chunks"):
    """
    Load an existing ChromaDB collection.
    
    Args:
        persist_dir (str): Directory containing the collection
        collection_name (str): Name of the collection to load
    
    Returns:
        chromadb.Collection: Collection object, or None if failed
    """
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_collection(collection_name)
        print(f"✅ Loaded collection: {collection_name}")
        return collection
    except Exception as e:
        print(f"❌ Error loading collection: {e}")
        return None

def create_llm_agent(collection, model_name="gpt-3.5-turbo", temperature=0.1):
    """
    Create a LangChain agent with tools for querying the book.
    
    Args:
        collection: ChromaDB collection object
        model_name (str): Model to use (OpenRouter supported models)
        temperature (float): Model temperature for creativity control
    
    Returns:
        LangChain agent object
    """
    try:
        # Check for OpenRouter API key
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        
        if not openrouter_key:
            print("⚠️  Warning: OPENROUTER_API_KEY not found!")
            print("   Please set OPENROUTER_API_KEY in your .env file")
            print("   You can get a free OpenRouter key from: https://openrouter.ai/")
            return None
        
        # Initialize the LLM with OpenRouter
        print("🔑 Using OpenRouter API")
        llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            verbose=False,
            openai_api_key=openrouter_key,
            openai_api_base="https://openrouter.ai/api/v1"
        )
        
        # Create tools
        search_tool = ChromaDBSearchTool(collection)
        summary_tool = BookSummaryTool(collection)
        
        tools = [search_tool, summary_tool]
        
        # Create the agent with better configuration
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,  # Reduce verbose output
            handle_parsing_errors=True,
            max_iterations=3,  # Reduce iterations for faster responses
            agent_kwargs={
                "system_message": """You are a helpful AI assistant that has access to a book through search tools. 
                
                Your job is to:
                1. Use the search tools to find relevant information from the book
                2. Read and understand the content returned by the tools
                3. Synthesize the information into a clear, comprehensive answer
                4. Write your answer in your own words, not just repeat the search results
                5. If you can't find specific information, say so clearly
                
                Always provide thoughtful, well-structured answers based on the book content you find."""
            }
        )
        
        return agent
        
    except Exception as e:
        print(f"❌ Error creating LLM agent: {e}")
        return None

def ask_question_with_llm(agent, question: str) -> str:
    """
    Ask a question using the LLM agent.
    
    Args:
        agent: LangChain agent object
        question (str): Question to ask
    
    Returns:
        str: LLM's answer
    """
    try:
        if not agent:
            return "❌ LLM agent not available. Please check your OpenRouter API key and try again."
        
        # Add context about the book
        enhanced_question = f"""
        Please answer the following question about the book content:
        
        Question: {question}
        
        Important: Use the search tools to find relevant information, then synthesize it into a clear, 
        comprehensive answer in your own words. Do not just list the search results - provide a 
        thoughtful response that directly answers the question based on the book content you find.
        """
        
        response = agent.run(enhanced_question)
        return response
        
    except Exception as e:
        return f"❌ Error getting LLM response: {str(e)}"

def interactive_llm_query(collection, max_results=5):
    """
    Interactive query interface using LLM for intelligent answers.
    
    Args:
        collection: ChromaDB collection object
        max_results (int): Maximum number of search results to show
    """
    if not collection:
        print("❌ No collection provided")
        return
    
    # Show collection info
    info = get_collection_info(collection)
    if info:
        print(f"\n📚 Book Information:")
        print(f"  Collection: {info['name']}")
        print(f"  Total chunks: {info['count']}")
        print(f"  Created: {info['metadata'].get('created', 'Unknown')}")
    
    print(f"\n🤖 LLM-Powered Query Interface")
    print(f"Ask questions and get intelligent answers based on the book content!")
    print(f"Type your questions (type 'exit' to quit)")
    print("-" * 60)
    
    # Create LLM agent
    print("\n🔧 Initializing LLM agent...")
    agent = create_llm_agent(collection)
    
    if not agent:
        print("❌ Failed to create LLM agent. Falling back to basic search...")
        # Fall back to basic search
        while True:
            try:
                query = input("\n❓ Question: ").strip()
                
                if query.lower() in ['exit', 'quit', 'q', '']:
                    print("👋 Goodbye!")
                    break
                
                if len(query) < 3:
                    print("⚠️  Please enter a longer question (at least 3 characters)")
                    continue
                
                print(f"🔍 Searching for: '{query}'")
                results = query_chroma_collection(collection, query, n_results=max_results)
                
                if results and results['documents'] and results['documents'][0]:
                    documents = results['documents'][0]
                    metadatas = results['metadatas'][0]
                    
                    print(f"\n✅ Found {len(documents)} relevant sections:")
                    for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
                        page = metadata.get('page', 'Unknown')
                        print(f"\n--- Result {i+1} (Page {page}) ---")
                        print(f"Text: {doc[:300]}{'...' if len(doc) > 300 else ''}")
                else:
                    print("❌ No relevant results found.")
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error during fallback search: {e}")
                print("💡 Try a different question or restart the interface.")
        
        return
    
    print("✅ LLM agent ready!")
    
    while True:
        try:
            question = input("\n❓ Question: ").strip()
            
            if question.lower() in ['exit', 'quit', 'q', '']:
                print("👋 Goodbye!")
                break
            
            if len(question) < 3:
                print("⚠️  Please enter a longer question (at least 3 characters)")
                continue
            
            print(f"\n🤖 Thinking...")
            
            # Get LLM answer
            answer = ask_question_with_llm(agent, question)
            
            print(f"\n💡 Answer:")
            print(answer)
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error during query: {e}")
            print("💡 Try a different question or restart the interface.")

def search_book(collection, query_text, n_results=5):
    """
    Simple search function for programmatic use.
    
    Args:
        collection: ChromaDB collection object
        query_text (str): Text to search for
        n_results (int): Number of results to return
    
    Returns:
        dict: Search results or None if failed
    """
    if not collection:
        print("❌ No collection provided")
        return None
    
    try:
        results = query_chroma_collection(collection, query_text, n_results=n_results)
        return results
    except Exception as e:
        print(f"❌ Search error: {e}")
        return None

def get_book_summary(collection, n_results=10):
    """
    Get a summary of the book by querying for general content.
    
    Args:
        collection: ChromaDB collection object
        n_results (int): Number of chunks to include in summary
    
    Returns:
        dict: Summary results or None if failed
    """
    if not collection:
        print("❌ No collection provided")
        return None
    
    try:
        # Query for general book content
        results = query_chroma_collection(collection, "book content overview introduction", n_results=n_results)
        return results
    except Exception as e:
        print(f"❌ Summary error: {e}")
        return None

if __name__ == "__main__":
    # Direct access to interactive query interface
    print("🚀 LLM Query Interface")
    print("=" * 30)
    
    # Try to load a collection
    collection = load_collection()
    
    if collection:
        # Start interactive mode directly
        interactive_llm_query(collection)
    else:
        print("❌ No collection found. Please run the main pipeline first to create embeddings.")
