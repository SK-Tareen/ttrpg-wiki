from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from pdfminer.high_level import extract_text

pdf_path = "name.pdf"
text = extract_text(pdf_path)

with open("book.txt", "w", encoding="utf-8") as f:
    f.write(text)

# Load book
loader = TextLoader("book.txt")
documents = loader.load()

# Chunk book into manageable pieces
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(documents)

# Create embeddings (you can use 'all-MiniLM-L6-v2' locally)
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Store vectors in Chroma
vectordb = Chroma.from_documents(docs, embedding, persist_directory="./book_chroma")
vectordb.persist()
