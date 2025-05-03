import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일 불러오기

# 기본 환경 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 디렉토리 설정
DATA_DIR = "../data/manuals"
CHROMA_DIR = "../db/chroma"

# 모델 설정
LLM_MODEL = "gpt-3.5-turbo"
LLM_TEMPERATURE = 0
