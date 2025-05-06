import os
import time
from tqdm import tqdm
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings
from app.config import DATA_DIR, CHROMA_DIR, OPENAI_API_KEY


def load_pdfs_and_create_vectorstore(pdf_paths):
    print("🚀 시작합니다...")
    documents = []

    # PDF 로드
    for filename in pdf_paths:
        if filename.endswith(".pdf"):
            filepath = os.path.join(DATA_DIR, filename)
            print(f"📄 Loading: {filename}")
            loader = PyMuPDFLoader(filepath)
            docs = loader.load()
            documents.extend(docs)

    # 텍스트 분할
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    chunks = [chunk for chunk in chunks if chunk.page_content and len(chunk.page_content.strip()) > 20]
    print(f"🧩 Total valid chunks created: {len(chunks)}")

    # Chroma 임베딩 및 저장소 설정
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    client_settings = Settings(anonymized_telemetry=False, allow_reset=True)
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
        client_settings=client_settings
    )

    # 배치 단위 임베딩 및 저장
    batch_size = 50
    for i in tqdm(range(0, len(chunks), batch_size), desc="📥 Embedding batches"):
        batch = chunks[i:i+batch_size]
        print(f"\n🌀 Embedding batch {i//batch_size + 1} / {(len(chunks)-1)//batch_size + 1}")
        print(f"⏳ Embedding 시작 중...")
        try:
            vectordb.add_documents(batch)
            print(f"✅ Batch {i//batch_size + 1} 저장 성공 (⏱ {round(time.perf_counter(), 2)}초)")
            time.sleep(1.2)  # API 부하 방지
        except Exception as e:
            print(f"❌ Batch {i//batch_size + 1} 실패: {type(e).__name__} - {e}")

    vectordb.persist()
    print(f"✅ 전체 저장 완료 → {CHROMA_DIR}")


if __name__ == "__main__":
    all_pdfs = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    total = len(all_pdfs)
    batch_count = (total - 1) // 20 + 1

    print("🚀 전체 PDF를 배치로 벡터화 시작합니다...")
    for i in range(batch_count):
        batch = all_pdfs[i*20:(i+1)*20]
        print(f"\n🔄 Batch {i+1} / {batch_count} 처리 중...")
        load_pdfs_and_create_vectorstore(batch)
        time.sleep(3)
