#!/usr/bin/env python3
"""
Enhanced LLM-powered query system for your book.
Ask questions and get intelligent answers based on your book content.
"""

import os
from dotenv import load_dotenv
from utils.query import load_collection, interactive_llm_query

# Load environment variables from .env file
load_dotenv()

def main():
    print("🚀 LLM-Powered Book Query System")
    print("=" * 50)
    
    # Check for OpenRouter API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Error: OPENROUTER_API_KEY not found!")
        print("Please set your OpenRouter API key in your .env file:")
        print("  OPENROUTER_API_KEY=your-key-here")
        print("\nYou can get a free API key from: https://openrouter.ai/")
        return
    
    # Load the ChromaDB collection
    print("\n📚 Loading book collection...")
    collection = load_collection()
    
    if not collection:
        print("❌ Failed to load collection. Make sure you've run the main pipeline first.")
        return
    
    print("✅ Collection loaded successfully!")
    
    # Go straight to interactive query mode
    print("\n🚀 Starting interactive query mode...")
    print("Ask questions about your book content!")
    print("Type 'exit' to quit.")
    print("-" * 50)
    
    interactive_llm_query(collection)

if __name__ == "__main__":
    main()
