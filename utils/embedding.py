import logging
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
import os

def embed_and_store_chroma(chunks, persist_dir="chroma_db", batch_size=16):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    os.makedirs(persist_dir, exist_ok=True)
    client = PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection("rag_chunks")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    logging.info(f"Loaded embedding model. Total chunks to process: {len(chunks)}")

    valid_chunks = []
    ids = []
    for j, chunk in enumerate(chunks):
        if isinstance(chunk["text"], str) and chunk["text"].strip():
            doc_id = f"page-{chunk['page']}-chunk-{j}"
            ids.append(doc_id)
            valid_chunks.append(chunk)

    logging.info(f"Prepared {len(valid_chunks)} valid chunks.")

    existing_ids = set(collection.get(ids=ids)["ids"])
    logging.info(f"Found {len(existing_ids)} existing embeddings. Skipping those.")

    chunks_to_embed = [chunk for j, chunk in enumerate(valid_chunks) 
                       if f"page-{chunk['page']}-chunk-{j}" not in existing_ids]
    new_ids = [f"page-{chunk['page']}-chunk-{j}" for j, chunk in enumerate(valid_chunks) 
               if f"page-{chunk['page']}-chunk-{j}" not in existing_ids]

    for i in range(0, len(chunks_to_embed), batch_size):
        batch = chunks_to_embed[i:i+batch_size]
        batch_ids = new_ids[i:i+batch_size]
        texts = [chunk["text"] for chunk in batch]
        metadatas = [{"page": chunk["page"]} for chunk in batch]  # Metadata here

        try:
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            collection.add(
                documents=texts,
                embeddings=embeddings.tolist(),
                ids=batch_ids,
                metadatas=metadatas  # <-- metadata added here
            )
            logging.info(f"Embedded and stored batch {i // batch_size + 1} with {len(batch)} new chunks.")
        except Exception as e:
            logging.error(f"Error embedding batch {i // batch_size + 1}: {e}")
    
    client.persist()

    logging.info("✅ All done. Vector DB is saved and reusable.")
