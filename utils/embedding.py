import logging
import os
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def embed_and_store_faiss(chunks, faiss_index_path="faiss_index.index", metadata_path="faiss_metadata.pkl", batch_size=16):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    logging.info(f"Loaded embedding model. Total chunks to process: {len(chunks)}")

    valid_chunks = []
    ids = []
    texts = []
    metadatas = []
    
    for j, chunk in enumerate(chunks):
        if isinstance(chunk["text"], str) and chunk["text"].strip():
            doc_id = f"page-{chunk['page']}-chunk-{j}"
            ids.append(doc_id)
            texts.append(chunk["text"])
            metadatas.append({"page": chunk["page"], "id": doc_id})
            valid_chunks.append(chunk)

    logging.info(f"Prepared {len(valid_chunks)} valid chunks.")

    # Check if FAISS index and metadata already exist
    if os.path.exists(faiss_index_path) and os.path.exists(metadata_path):
        index = faiss.read_index(faiss_index_path)
        with open(metadata_path, "rb") as f:
            all_metadata = pickle.load(f)
    else:
        index = faiss.IndexFlatL2(model.get_sentence_embedding_dimension())
        all_metadata = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_metadata = metadatas[i:i+batch_size]
        try:
            embeddings = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
            index.add(np.array(embeddings, dtype=np.float32))
            all_metadata.extend(batch_metadata)
            logging.info(f"Embedded and stored batch {i // batch_size + 1} with {len(batch_texts)} chunks.")
        except Exception as e:
            logging.error(f"Error embedding batch {i // batch_size + 1}: {e}")

    # Save FAISS index and metadata
    faiss.write_index(index, faiss_index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(all_metadata, f)

    logging.info("✅ All done. FAISS index and metadata saved.")
