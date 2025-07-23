from utils.query import load_collection, query_book, ask_llm

def main():
    collection = load_collection("chroma_db")
    while True:
        user_query = input("Enter your question (or 'exit' to quit): ").strip()
        if user_query.lower() == "exit":
            break
        
        # Query the vector DB for relevant chunks
        context_chunks = query_book(collection, user_query, k=5)
        
        if not context_chunks:
            print("No relevant content found.")
            continue
        
        # Ask the LLM using retrieved chunks as context
        answer = ask_llm(user_query, context_chunks)
        
        print("\nAnswer:")
        print(answer)
        print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    main()
