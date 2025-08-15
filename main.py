from utils.parse_pdf import parse_pdf, validate_pdf_file
from utils.chunking import chunk_book_from_dict, validate_chunks, print_chunk_statistics
from utils.embedding import embed_and_store_chroma, query_chroma_collection, get_collection_info

def main():
    book_path = "book.pdf"
    
    # Step 1: Parse PDF
    print("="*60)
    print("STEP 1: Parsing PDF with PyMuPDF")
    print("="*60)
    
    is_valid, message = validate_pdf_file(book_path)
    if is_valid:
        result = parse_pdf(book_path, "book.json", max_workers=4)
    else:
        print(f"PDF validation failed: {message}")
        return

    # Step 2: Chunk
    if result:
        print("\n" + "="*60)
        print("STEP 2: Creating text chunks")
        print("="*60)
        
        # Pass the result dictionary directly to chunking
        chunks = chunk_book_from_dict(result, chunk_size=500, chunk_overlap=50)
        
        if chunks:
            # Validate and show statistics
            validation = validate_chunks(chunks)
            print_chunk_statistics(chunks)
            
            print(f"\n✅ Successfully created {len(chunks)} chunks!")
        else:
            print("❌ Chunking failed!")
            return
    else:
        print("❌ PDF parsing failed!")
        return

    # Step 3: Create embeddings and store in ChromaDB
    if chunks:
        print("\n" + "="*60)
        print("STEP 3: Creating embeddings and storing in ChromaDB")
        print("="*60)
        
        collection = embed_and_store_chroma(chunks, persist_dir="chroma_db", collection_name="book_chunks")
        
        if collection:
            print("✅ Embeddings stored successfully!")
            
            # Show collection info
            info = get_collection_info(collection)
            if info:
                print(f"\nCollection Information:")
                print(f"  Name: {info['name']}")
                print(f"  Document count: {info['count']}")
                print(f"  Created: {info['metadata'].get('created', 'Unknown')}")
            
            # Test a simple query
            print(f"\n🧪 Testing query system...")
            test_query = "What is this book about?"
            results = query_chroma_collection(collection, test_query, n_results=3)
            
            if results and results['documents']:
                print(f"✅ Query test successful! Found {len(results['documents'][0])} results")
                print(f"First result preview: {results['documents'][0][0][:100]}...")
            else:
                print("⚠️  Query test completed but no results returned")
            
        else:
            print("❌ Embedding failed!")
            return
    else:
        print("❌ No chunks to embed!")
        return

    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()
