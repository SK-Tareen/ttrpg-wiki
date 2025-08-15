import time
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_book_from_dict(book_data, chunk_size=500, chunk_overlap=50, show_progress=True):
    """
    Split book text into overlapping chunks from a dictionary (not JSON file).
    
    Args:
        book_data (dict): Dictionary containing book data (page -> text)
        chunk_size (int): Maximum size of each chunk in characters
        chunk_overlap (int): Number of characters to overlap between chunks
        show_progress (bool): Whether to show progress indicators
    
    Returns:
        list: List of dictionaries containing chunk information
    """
    
    # Start timing
    start_time = time.time()
    start_datetime = datetime.now()
    
    print(f"=== Text Chunking Started at {start_datetime.strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    # Validate input
    if not isinstance(book_data, dict):
        print("Error: book_data must be a dictionary")
        return None
    
    total_pages = len(book_data)
    if total_pages == 0:
        print("Error: No pages found in book_data")
        return None
    
    print(f"Processing {total_pages} pages from dictionary")
    print(f"Chunk size: {chunk_size} characters, Overlap: {chunk_overlap} characters")
    
    # Validate chunk parameters
    if chunk_size <= 0:
        print("Error: Chunk size must be positive")
        return None
    
    if chunk_overlap < 0:
        print("Error: Chunk overlap cannot be negative")
        return None
    
    if chunk_overlap >= chunk_size:
        print("Error: Chunk overlap must be less than chunk size")
        return None
    
    try:
        # Initialize text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        all_chunks = []
        processed_pages = 0
        skipped_pages = 0
        total_chunks = 0
        
        print(f"Starting text chunking process...")
        
        # Process each page
        for page_num, content in book_data.items():
            processed_pages += 1
            
            # Progress indicator
            if show_progress:
                progress = (processed_pages / total_pages) * 100
                print(f"\rProcessing page {processed_pages}/{total_pages} ({progress:.1f}%)", end="", flush=True)
            
            # Skip error pages
            if "[Error:" in content:
                skipped_pages += 1
                continue
            
            # Skip empty content
            if not content or not content.strip():
                skipped_pages += 1
                continue
            
            try:
                # Split text into chunks
                chunks = splitter.split_text(content)
                
                # Add chunk metadata
                for idx, chunk in enumerate(chunks):
                    if chunk.strip():  # Only add non-empty chunks
                        all_chunks.append({
                            "page": page_num,
                            "chunk_id": f"{page_num}_{idx}",
                            "text": chunk,
                            "chunk_size": len(chunk),
                            "chunk_index": idx
                        })
                        total_chunks += 1
                
                # Periodic progress updates
                if processed_pages % 10 == 0 or processed_pages == total_pages:
                    print(f"\nProcessed {processed_pages}/{total_pages} pages - Created {total_chunks} chunks so far...")
            
            except Exception as e:
                print(f"\nWarning: Could not process page {page_num}: {e}")
                skipped_pages += 1
                continue
        
        # Clear progress line
        if show_progress:
            print()  # New line after progress
        
        # Calculate timing
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Final statistics
        print(f"\n=== Text Chunking Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        print(f"Total processing time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        print(f"Pages processed: {processed_pages}")
        print(f"Pages skipped: {skipped_pages}")
        print(f"Total chunks created: {total_chunks}")
        
        # Calculate performance metrics
        if total_pages > 0 and total_duration > 0:
            pages_per_second = total_pages / total_duration
            chunks_per_second = total_chunks / total_duration
            print(f"Processing speed: {pages_per_second:.2f} pages/second")
            print(f"Chunking speed: {chunks_per_second:.2f} chunks/second")
        
        # Validate results
        if total_chunks == 0:
            print("Error: No chunks were created")
            return None
        
        # Success summary
        success_rate = (processed_pages - skipped_pages) / total_pages * 100
        print(f"Success rate: {success_rate:.1f}%")
        
        if skipped_pages > 0:
            print(f"Warning: {skipped_pages} pages were skipped due to errors or empty content")
        
        return all_chunks
        
    except Exception as e:
        print(f"Error: Unexpected error during chunking: {e}")
        return None

def validate_chunks(chunks):
    """
    Validate the quality and structure of created chunks.
    
    Args:
        chunks (list): List of chunk dictionaries
        
    Returns:
        dict: Validation results and statistics
    """
    if not chunks:
        return {"valid": False, "error": "No chunks provided"}
    
    validation_results = {
        "valid": True,
        "total_chunks": len(chunks),
        "empty_chunks": 0,
        "very_small_chunks": 0,
        "very_large_chunks": 0,
        "missing_fields": 0,
        "avg_chunk_size": 0,
        "min_chunk_size": float('inf'),
        "max_chunk_size": 0
    }
    
    total_size = 0
    
    for chunk in chunks:
        # Check required fields
        required_fields = ["page", "chunk_id", "text"]
        if not all(field in chunk for field in required_fields):
            validation_results["missing_fields"] += 1
            validation_results["valid"] = False
        
        # Check chunk size
        chunk_size = len(chunk.get("text", ""))
        total_size += chunk_size
        
        if chunk_size == 0:
            validation_results["empty_chunks"] += 1
            validation_results["valid"] = False
        elif chunk_size < 10:
            validation_results["very_small_chunks"] += 1
        elif chunk_size > 2000:
            validation_results["very_large_chunks"] += 1
        
        # Update min/max
        validation_results["min_chunk_size"] = min(validation_results["min_chunk_size"], chunk_size)
        validation_results["max_chunk_size"] = max(validation_results["max_chunk_size"], chunk_size)
    
    # Calculate averages
    if validation_results["total_chunks"] > 0:
        validation_results["avg_chunk_size"] = total_size / validation_results["total_chunks"]
    
    return validation_results

def print_chunk_statistics(chunks):
    """
    Print detailed statistics about the created chunks.
    
    Args:
        chunks (list): List of chunk dictionaries
    """
    if not chunks:
        print("No chunks to analyze")
        return
    
    print("\n=== Chunk Statistics ===")
    print(f"Total chunks: {len(chunks)}")
    
    # Page distribution
    pages = set(chunk["page"] for chunk in chunks)
    print(f"Pages covered: {len(pages)}")
    
    # Size distribution
    sizes = [len(chunk["text"]) for chunk in chunks]
    sizes.sort()
    
    print(f"Chunk sizes:")
    print(f"  Smallest: {sizes[0]} characters")
    print(f"  Largest: {sizes[-1]} characters")
    print(f"  Median: {sizes[len(sizes)//2]} characters")
    print(f"  Average: {sum(sizes)/len(sizes):.1f} characters")
    
    # Quality indicators
    empty_chunks = sum(1 for chunk in chunks if not chunk["text"].strip())
    if empty_chunks > 0:
        print(f"  Empty chunks: {empty_chunks}")
    
    very_small = sum(1 for size in sizes if size < 50)
    if very_small > 0:
        print(f"  Very small chunks (<50 chars): {very_small}")
    
    very_large = sum(1 for size in sizes if size > 1000)
    if very_large > 0:
        print(f"  Very large chunks (>1000 chars): {very_large}")
