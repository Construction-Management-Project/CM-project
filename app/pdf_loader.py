import os
from tqdm import tqdm
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from app.config import DATA_DIR, CHROMA_DIR, OPENAI_API_KEY

def load_pdfs_and_create_vectorstore(pdf_paths):
    print("🚀 Starting vectorization...")
    documents = []

    # Load PDFs
    for filename in pdf_paths:
        if filename.endswith(".pdf"):
            filepath = os.path.join(DATA_DIR, filename)
            print(f"📄 Loading: {filename}")
            loader = PyMuPDFLoader(filepath)
            docs = loader.load()
            documents.extend(docs)

    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    print(f"🧩 Total valid chunks created: {len(chunks)}")

    # Embed and save to FAISS
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vectordb.save_local(CHROMA_DIR)
    print(f"✅ FAISS vector store saved to → {CHROMA_DIR}")

if __name__ == "__main__":
    all_pdfs = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    total = len(all_pdfs)
    batch_count = (total - 1) // 20 + 1

    print("🚀 Starting batch processing of all PDFs...")
    for i in range(batch_count):
        batch = all_pdfs[i * 20:(i + 1) * 20]
        print(f"\n🔄 Processing batch {i + 1} / {batch_count}...")
        load_pdfs_and_create_vectorstore(batch)
