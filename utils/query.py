import chromadb
from chromadb.config import Settings
import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"

def load_collection(persist_dir="chroma_db"):
    settings = Settings(
        persist_directory=persist_dir
    )
    client = chromadb.Client(settings=settings)
    return client.get_collection("rag_chunks")

def query_book(collection, query, k=5):
    results = collection.query(query_texts=[query], n_results=k)
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": doc, "page": meta["page"]})
    return chunks

def ask_llm(query, context_chunks):
    context = "\n\n".join([f"[Page {c['page']}]: {c['text']}" for c in context_chunks])
    prompt = f"Answer the question based on the following book content:\n\n{context}\n\nQuestion: {query}"

    response = openai.ChatCompletion.create(
        model="mistralai/mixtral-8x7b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who answers based on book content."},
            {"role": "user", "content": prompt}
        ]
    )
    return response["choices"][0]["message"]["content"]
