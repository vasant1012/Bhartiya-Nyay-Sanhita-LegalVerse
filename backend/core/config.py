from functools import lru_cache
from pathlib import Path

import dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

dotenv.load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')
    app_name: str = 'LegalVerse API'
    app_version: str = '0.1.0'
    app_env: str = 'development'
    data_file: Path = Path( dotenv.get_key(dotenv.find_dotenv(), 'DATA_FILE'))
    processed_data_file: Path = Path(dotenv.get_key(dotenv.find_dotenv(), 'PROCESSED_DATA_FILE'))
    artifacts_dir: Path = Path(dotenv.get_key(dotenv.find_dotenv(), 'ARTIFACTS_DIR'))
    faiss_index_file: Path = Path(dotenv.get_key(dotenv.find_dotenv(), 'FAISS_INDEX_FILE'))
    faiss_metadata_file: Path = Path(dotenv.get_key(dotenv.find_dotenv(), 'FAISS_METADATA_FILE'))
    embedding_model: str = dotenv.get_key(dotenv.find_dotenv(), 'EMBEDDING_MODEL')
    neo4j_uri: str = Field(default=dotenv.get_key(dotenv.find_dotenv(), 'NEO4J_URI'))
    neo4j_username: str = Field(default=dotenv.get_key(dotenv.find_dotenv(), 'NEO4J_USERNAME'))
    neo4j_password: str = Field(default=dotenv.get_key(dotenv.find_dotenv(), 'NEO4J_PASSWORD'))
    neo4j_database: str = Field(default=dotenv.get_key(dotenv.find_dotenv(), 'NEO4J_DATABASE'))
    enable_faiss: bool = dotenv.get_key(dotenv.find_dotenv(), 'ENABLE_FAISS') == 'true'
    enable_neo4j: bool = dotenv.get_key(dotenv.find_dotenv(), 'ENABLE_NEO4J') == 'true'
    search_top_k: int = Field(default=int(dotenv.get_key(dotenv.find_dotenv(), 'SEARCH_TOP_K')), ge=1, le=50)

@lru_cache
def get_settings() -> Settings:
    return Settings()
