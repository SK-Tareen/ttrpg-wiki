import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_book(json_path, chunk_size=500, chunk_overlap=50):
    with open(json_path, "r", encoding="utf-8") as f:
        book_data = json.load(f)

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []

    for page, content in book_data.items():
        if "[Error:" in content:
            continue
        chunks = splitter.split_text(content)
        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "page": page,
                "chunk_id": f"{page}_{idx}",
                "text": chunk
            })

    print(f"Total chunks created: {len(all_chunks)}")  # <-- Added this line
    return all_chunks
