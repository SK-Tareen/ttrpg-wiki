import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import time
from datetime import datetime

def embed_and_store_chroma(chunks, persist_dir="chroma_db", collection_name="book_chunks", show_progress=True):
    """
    Safest and easiest embedding solution using ChromaDB.
    
    Args:
        chunks (list): List of chunk dictionaries from chunking
        persist_dir (str): Directory to persist ChromaDB data
        collection_name (str): Name of the collection to create/use
        show_progress (bool): Whether to show progress indicators
    
    Returns:
        chromadb.Collection: ChromaDB collection object, or None if failed
    """
    
    start_time = time.time()
    start_datetime = datetime.now()
    
    print(f"=== ChromaDB Embedding Started at {start_datetime.strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    # Input validation
    if not chunks:
        print("Error: No chunks provided")
        return None
    
    if not isinstance(chunks, list):
        print("Error: chunks must be a list")
        return None
    
    print(f"Processing {len(chunks)} chunks...")
    
    try:
        # Initialize ChromaDB
        print("Initializing ChromaDB...")
        client = chromadb.PersistentClient(path=persist_dir)
        
        # Create or get collection
        try:
            collection = client.create_collection(
                name=collection_name,
                metadata={"description": "Book chunks with embeddings", "created": start_datetime.isoformat()}
            )
            print(f"✅ Created new collection: {collection_name}")
        except Exception as e:
            # Collection might already exist
            try:
                collection = client.get_collection(collection_name)
                print(f"✅ Using existing collection: {collection_name}")
                print(f"   Current documents: {collection.count()}")
            except Exception as e2:
                print(f"Error accessing collection: {e2}")
                return None
        
        # Load embedding model
        print("Loading embedding model (all-MiniLM-L6-v2)...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"✅ Model loaded successfully")
        
        # Prepare data
        print("Preparing chunks for embedding...")
        documents = []
        metadatas = []
        ids = []
        valid_chunks = 0
        
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, dict) and "text" in chunk and chunk["text"].strip():
                documents.append(chunk["text"])
                metadatas.append({
                    "page": chunk.get("page", "unknown"),
                    "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
                    "chunk_size": chunk.get("chunk_size", len(chunk["text"])),
                    "chunk_index": chunk.get("chunk_index", i),
                    "source": "pdf_parser"
                })
                ids.append(f"chunk_{i:06d}")
                valid_chunks += 1
                
                # Progress indicator
                if show_progress and (i + 1) % 100 == 0:
                    print(f"  Prepared {i + 1}/{len(chunks)} chunks...")
        
        print(f"✅ Prepared {valid_chunks} valid chunks for embedding")
        
        if valid_chunks == 0:
            print("Error: No valid chunks found")
            return None
        
        # Add documents to collection (ChromaDB handles embeddings automatically)
        print("Adding documents to ChromaDB collection...")
        print("Note: ChromaDB will automatically create embeddings for each document")
        
        # Process in batches to avoid memory issues
        batch_size = 100
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in range(0, len(documents), batch_size):
            batch_num = i // batch_size + 1
            end_idx = min(i + batch_size, len(documents))
            
            batch_docs = documents[i:end_idx]
            batch_metas = metadatas[i:end_idx]
            batch_ids = ids[i:end_idx]
            
            if show_progress:
                print(f"\rProcessing batch {batch_num}/{total_batches} ({len(batch_docs)} chunks)", end="", flush=True)
            
            try:
                collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
            except Exception as e:
                print(f"\nWarning: Error in batch {batch_num}: {e}")
                continue
        
        if show_progress:
            print()  # New line after progress
        
        # Verify storage
        final_count = collection.count()
        print(f"✅ Successfully stored {final_count} chunks in ChromaDB")
        
        # Collection info
        print(f"Collection name: {collection_name}")
        print(f"Persist directory: {persist_dir}")
        print(f"Embedding model: all-MiniLM-L6-v2")
        
        # Timing
        duration = time.time() - start_time
        print(f"Total processing time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        if final_count > 0 and duration > 0:
            chunks_per_second = final_count / duration
            print(f"Processing speed: {chunks_per_second:.2f} chunks/second")
        
        return collection
        
    except Exception as e:
        print(f"❌ Fatal error during embedding: {e}")
        return None

def query_chroma_collection(collection, query_text, n_results=5):
    """
    Query the ChromaDB collection for similar chunks.
    
    Args:
        collection: ChromaDB collection object
        query_text (str): Text to search for
        n_results (int): Number of results to return
    
    Returns:
        dict: Query results
    """
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results
    except Exception as e:
        print(f"Error querying collection: {e}")
        return None

def get_collection_info(collection):
    """
    Get information about the ChromaDB collection.
    
    Args:
        collection: ChromaDB collection object
    
    Returns:
        dict: Collection information
    """
    try:
        info = {
            "name": collection.name,
            "count": collection.count(),
            "metadata": collection.metadata
        }
        return info
    except Exception as e:
        print(f"Error getting collection info: {e}")
        return None

def delete_collection(persist_dir="chroma_db", collection_name="book_chunks"):
    """
    Delete a ChromaDB collection (use with caution!).
    
    Args:
        persist_dir (str): Directory containing the collection
        collection_name (str): Name of the collection to delete
    """
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        client.delete_collection(collection_name)
        print(f"✅ Deleted collection: {collection_name}")
    except Exception as e:
        print(f"Error deleting collection: {e}")

def list_collections(persist_dir="chroma_db"):
    """
    List all collections in the ChromaDB directory.
    
    Args:
        persist_dir (str): Directory to check
    
    Returns:
        list: List of collection names
    """
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        collections = client.list_collections()
        return [col.name for col in collections]
    except Exception as e:
        print(f"Error listing collections: {e}")
        return []
