import os
from dotenv import load_dotenv

load_dotenv()  # .env

# Setting
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Directory
DATA_DIR = os.environ.get("DATA_DIR", "data/manuals")
CHROMA_DIR = os.environ.get("CHROMA_DIR", "db/faiss_index")

# LLM
LLM_MODEL = "gpt-3.5-turbo"
LLM_TEMPERATURE = 0
