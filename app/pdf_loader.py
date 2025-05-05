import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm
from app.config import OPENAI_API_KEY, DATA_DIR, CHROMA_DIR

def load_pdfs_and_create_vectorstore():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    documents = []

    # PDF file load
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            filepath = os.path.join(DATA_DIR, filename)
            print(f"Loading: {filename}")
            loader = PyMuPDFLoader(filepath)
            docs = loader.load()
            documents.extend(docs)

    # text chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    # OpenAI Embedding
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # ChromaDB save
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    vectordb.persist()
    print(f"Vectorstore saved to: {CHROMA_DIR}")

if __name__ == "__main__":
    load_pdfs_and_create_vectorstore()



# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os

# PDF_DIR = "../data/manuals"

# def extract_and_chunk(pdf_path):
#     loader = PyMuPDFLoader(pdf_path)
#     documents = loader.load()

#     print(f"\n📄 PDF에서 추출된 페이지 수: {len(documents)}")

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=50
#     )
#     chunks = splitter.split_documents(documents)

#     print(f"🧩 생성된 청크 수: {len(chunks)}")
#     print(f"\n🔍 예시 청크 내용:\n{chunks[0].page_content[:300]}...\n")

# if __name__ == "__main__":
#     for filename in os.listdir(PDF_DIR):
#         if filename.endswith(".pdf"):
#             path = os.path.join(PDF_DIR, filename)
#             print(f"\n===============================")
#             print(f"📂 파일: {filename}")
#             extract_and_chunk(path)
