import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL") or "sqlite:///paperloom.db"
DATABASE_CHUNK_SIZE = int(os.getenv("DATABASE_CHUNK_SIZE") or 5_000)
MILVUS_DB_URL = os.getenv("VECTOR_DB_URL") or "sqlite:///milvus.db"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
