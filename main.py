from utils.parse_pdf import parse_pdf
from utils.chunking import chunk_book
from utils.embedding import embed_and_store_faiss
from utils.query import load_collection, query_book, ask_llm

def main():
    # Step 1: Parse PDF
    data = parse_pdf("book.pdf", output_json_path="book.json")
    if not data:
        print("Parsing failed.")
        return

    # Step 2: Chunk
    chunks = chunk_book("book.json")

    # Step 3: Embed + Store in Chroma
    embed_and_store_faiss(chunks, persist_dir="chroma_db", batch_size=16)

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
