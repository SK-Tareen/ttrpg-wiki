import fitz
import json
import os
import concurrent.futures
import warnings
import time
from datetime import datetime

# Suppress common PyMuPDF warnings that don't affect functionality
warnings.filterwarnings("ignore", message=".*Data-loss while decompressing.*")
warnings.filterwarnings("ignore", message=".*Could get FontBBox.*")
warnings.filterwarnings("ignore", message=".*font descriptor.*")

def parse_pdf(pdf_path, output_json_path="book.json", show_progress=True, max_workers=4):
    """
    Fast PDF parsing with PyMuPDF and parallel processing.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_json_path (str): Path for the output JSON file
        show_progress (bool): Whether to show progress indicators
        max_workers (int): Number of parallel workers for processing
    
    Returns:
        dict: Dictionary containing page text, or None if failed
    """
    
    # Start timing
    start_time = time.time()
    start_datetime = datetime.now()
    
    print(f"=== PDF Parsing Started at {start_datetime.strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    # Input validation and error handling
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        return None
    
    if not pdf_path.lower().endswith('.pdf'):
        print(f"Error: File is not a PDF: {pdf_path}")
        return None
    
    # Check file size
    file_size = os.path.getsize(pdf_path)
    if file_size == 0:
        print(f"Error: PDF file is empty: {pdf_path}")
        return None
    
    print(f"Starting fast PDF parsing: {pdf_path} (Size: {file_size / 1024 / 1024:.2f} MB)")
    print(f"Using {max_workers} parallel workers with PyMuPDF")
    
    pages_dict = {}
    total_pages = 0
    successful_pages = 0
    failed_pages = 0
    empty_pages = 0
    
    def process_page(page_data):
        """Process a single page with PyMuPDF"""
        page_num, page = page_data
        try:
            # Extract text using PyMuPDF
            text = page.get_text()
            
            if text and text.strip():
                return page_num, text, "success"
            else:
                return page_num, "[Empty page - no text content]", "empty"
                
        except MemoryError:
            return page_num, "[Error: Memory limit exceeded while processing this page]", "failed"
        except Exception as e:
            return page_num, f"[Error: {str(e)}]", "failed"
    
    try:
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
            print(f"PDF opened successfully. Total pages: {total_pages}")
            
            if total_pages == 0:
                print("Error: PDF contains no pages")
                return None
            
            # Create page data for parallel processing
            page_data = [(i + 1, doc[i]) for i in range(total_pages)]
            
            print(f"Processing {total_pages} pages in parallel with PyMuPDF...")
            
            # Process pages in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all page processing tasks
                future_to_page = {executor.submit(process_page, data): data for data in page_data}
                
                # Process completed tasks
                completed = 0
                for future in concurrent.futures.as_completed(future_to_page):
                    page_num, text, status = future.result()
                    pages_dict[str(page_num)] = text
                    
                    # Update counters
                    if status == "success":
                        successful_pages += 1
                    elif status == "empty":
                        empty_pages += 1
                    else:
                        failed_pages += 1
                    
                    completed += 1
                    
                    # Progress indicator
                    if show_progress:
                        progress = (completed / total_pages) * 100
                        print(f"\rProcessing: {completed}/{total_pages} ({progress:.1f}%)", end="", flush=True)
                    
                    # Periodic updates
                    if completed % 20 == 0 or completed == total_pages:
                        print(f"\nProcessed {completed}/{total_pages} pages...")
                    
                    # Memory management - clear completed futures
                    del future
    
    except MemoryError:
        print(f"\nError: Memory limit exceeded while processing PDF: {pdf_path}")
        return None
    except PermissionError:
        print(f"\nError: Permission denied accessing PDF: {pdf_path}")
        return None
    except Exception as e:
        print(f"\nError: Failed to process PDF '{pdf_path}': {e}")
        return None
    
    # Clear progress line
    if show_progress:
        print()  # New line after progress
    
    # Calculate timing
    end_time = time.time()
    end_datetime = datetime.now()
    total_duration = end_time - start_time
    
    # Final statistics with timing
    print(f"\n=== PDF Parsing Completed at {end_datetime.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"Total parsing time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print(f"Parsing complete!")
    print(f"Total pages: {total_pages}")
    print(f"Successful: {successful_pages}")
    print(f"Empty: {empty_pages}")
    print(f"Failed: {failed_pages}")
    
    # Calculate performance metrics
    if total_pages > 0 and total_duration > 0:
        pages_per_second = total_pages / total_duration
        print(f"Processing speed: {pages_per_second:.2f} pages/second")
    
    # Validate we have some content
    if successful_pages == 0:
        print("Error: No pages were successfully parsed")
        return None
    
    # Save as JSON with error handling
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_json_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        with open(output_json_path, "w", encoding="utf-8") as json_file:
            json.dump(pages_dict, json_file, ensure_ascii=False, indent=2)
        
        print(f"JSON saved successfully to: {output_json_path}")
        
        # Verify file was written
        if os.path.exists(output_json_path):
            output_size = os.path.getsize(output_json_path)
            print(f"Output file size: {output_size / 1024:.2f} KB")
        else:
            print("Error: Output file was not created")
            return None
            
    except PermissionError:
        print(f"Error: Permission denied writing to: {output_json_path}")
        return None
    except OSError as e:
        print(f"Error: OS error writing JSON: {e}")
        return None
    except Exception as e:
        print(f"Error: Failed to save JSON: {e}")
        return None
    
    # Success summary with final timing
    success_rate = (successful_pages / total_pages) * 100
    print(f"Success rate: {success_rate:.1f}%")
    
    if failed_pages > 0:
        print(f"Warning: Some pages failed to parse. Check the output for details.")
    
    # Final timing summary
    final_time = time.time()
    total_with_save = final_time - start_time
    print(f"\n=== Total time including file save: {total_with_save:.2f} seconds ({total_with_save/60:.2f} minutes) ===")
    
    return pages_dict

def validate_pdf_file(pdf_path):
    """
    Validate PDF file before processing.
    
    Args:
        pdf_path (str): Path to PDF file
        
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # Check if file exists
        if not os.path.exists(pdf_path):
            return False, "File does not exist"
        
        # Check if it's a file (not directory)
        if not os.path.isfile(pdf_path):
            return False, "Path is not a file"
        
        # Check file extension
        if not pdf_path.lower().endswith('.pdf'):
            return False, "File is not a PDF"
        
        # Check file size
        file_size = os.path.getsize(pdf_path)
        if file_size == 0:
            return False, "File is empty"
        
        # Check if file is readable
        if not os.access(pdf_path, os.R_OK):
            return False, "File is not readable"
        
        return True, "File is valid"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def get_optimal_workers():
    """
    Get optimal number of workers based on system capabilities.
    
    Returns:
        int: Recommended number of workers
    """
    import multiprocessing
    
    # Use CPU count but cap at reasonable number to avoid overwhelming system
    cpu_count = multiprocessing.cpu_count()
    optimal_workers = min(cpu_count, 8)  # Cap at 8 to avoid memory issues
    
    return optimal_workers

