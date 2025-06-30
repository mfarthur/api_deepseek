# core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    # Carrega variáveis do arquivo .env
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Configs do LLM
    LLM_API_KEY: str
    LLM_API_URL: str

    # Configs da Base Vetorial
    VECTOR_DB_PATH: str = "db/chroma_db"
    VECTOR_COLLECTION_NAME: str = "document_collection"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

# Usamos lru_cache para que as configurações sejam carregadas apenas uma vez
@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()