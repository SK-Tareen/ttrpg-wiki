#!/usr/bin/env python3
"""
Example script demonstrating the enhanced LLM-powered query system.
This script shows how to use the LLM agent with tools to answer questions about your book.
"""

import os
from utils.query import load_collection, create_llm_agent, ask_question_with_llm, interactive_llm_query

def main():
    print("🚀 Enhanced LLM Query System Example")
    print("=" * 50)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found!")
        print("Please set your OpenAI API key:")
        print("  Windows: set OPENAI_API_KEY=your-key-here")
        print("  Linux/Mac: export OPENAI_API_KEY=your-key-here")
        print("\nYou can get an API key from: https://platform.openai.com/api-keys")
        return
    
    # Load the ChromaDB collection
    print("\n📚 Loading book collection...")
    collection = load_collection()
    
    if not collection:
        print("❌ Failed to load collection. Make sure you've run the main pipeline first.")
        return
    
    print("✅ Collection loaded successfully!")
    
    # Create LLM agent
    print("\n🤖 Creating LLM agent...")
    agent = create_llm_agent(collection, model_name="gpt-3.5-turbo", temperature=0.1)
    
    if not agent:
        print("❌ Failed to create LLM agent.")
        return
    
    print("✅ LLM agent created successfully!")
    
    # Example questions to demonstrate the system
    example_questions = [
        "What is this book about?",
        "What are the main topics covered?",
        "Can you summarize the key points?",
        "What are the most important concepts discussed?",
        "Are there any specific examples or case studies mentioned?"
    ]
    
    print(f"\n🧪 Testing with {len(example_questions)} example questions...")
    print("-" * 50)
    
    for i, question in enumerate(example_questions, 1):
        print(f"\n❓ Question {i}: {question}")
        print("🤖 Thinking...")
        
        try:
            answer = ask_question_with_llm(agent, question)
            print(f"💡 Answer: {answer}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 30)
    
    # Start interactive mode
    print(f"\n🚀 Starting interactive mode...")
    print("You can now ask your own questions!")
    print("Type 'exit' to quit.")
    
    interactive_llm_query(collection)

if __name__ == "__main__":
    main()
