from utils.parse_pdf import parse_pdf, validate_pdf_file

def main():
    book_path="book.pdf"
    # Step 1: Parse PDF
    is_valid, message = validate_pdf_file(book_path)
    if is_valid:
        result = parse_pdf(book_path, "book.json", max_workers=4)
    else:
        print(f"PDF validation failed: {message}")
        return

    # # Step 2: Chunk
    # chunks = chunk_book("book.json")

    # # Step 3: Embed + Store in Chroma
    # embed_and_store_faiss(chunks, persist_dir="chroma_db", batch_size=16)

    # # Step 4: Query Loop
    # collection = load_collection("chroma_db")
    # print("RAG system ready. Ask questions about the book.")
    # while True:
    #     q = input(">>> ")
    #     if q.lower() in {"exit", "quit"}:
    #         break
    #     context = query_book(collection, q)
    #     answer = ask_llm(q, context)
    #     print("\nAnswer:\n", answer)

if __name__ == "__main__":
    main()
