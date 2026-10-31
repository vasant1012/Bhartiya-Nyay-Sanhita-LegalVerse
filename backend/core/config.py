from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')
    app_name: str = 'LegalVerse API'
    app_version: str = '0.1.0'
    app_env: str = 'development'
    data_file: Path = Path('../data/bns_sections.csv')
    processed_data_file: Path = Path('../data/bns_processed.csv')
    artifacts_dir: Path = Path('../artifacts')
    faiss_index_file: Path = Path('../artifacts/faiss_indexes/bns_sections.index')
    faiss_metadata_file: Path = Path('../artifacts/faiss_indexes/bns_sections.json')
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    neo4j_uri: str = 'bolt://localhost:7687'
    neo4j_username: str = 'neo4j'
    neo4j_password: str = ''
    neo4j_database: str = 'neo4j'
    enable_faiss: bool = True
    enable_neo4j: bool = True
    search_top_k: int = Field(default=10, ge=1, le=50)

@lru_cache
def get_settings() -> Settings:
    return Settings()
