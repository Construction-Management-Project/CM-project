import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from tqdm import tqdm
from app.config import OPENAI_API_KEY, DATA_DIR, CHROMA_DIR

def load_pdfs_and_create_vectorstore():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    documents = []

    # PDF 파일들 로드
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            filepath = os.path.join(DATA_DIR, filename)
            print(f"Loading: {filename}")
            loader = PyMuPDFLoader(filepath)
            docs = loader.load()
            documents.extend(docs)

    # 텍스트 chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    # OpenAI 임베딩 생성
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Chroma DB 저장
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    vectordb.persist()
    print(f"Vectorstore saved to: {CHROMA_DIR}")





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
