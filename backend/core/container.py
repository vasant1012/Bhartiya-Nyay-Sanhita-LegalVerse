from backend.core.config import get_settings
from backend.intelligence.faiss_store import FaissStore
from backend.intelligence.neo4j_store import Neo4jStore
from backend.repositories.csv_repository import CsvSectionRepository
from backend.services.insight_service import InsightService
from backend.services.search_service import SearchService


class AppContainer:
    def __init__(self):
        s = get_settings()
        self.settings = s
        self.sections = CsvSectionRepository(s.data_file)
        self.faiss = FaissStore(s.faiss_index_file, s.faiss_metadata_file, s.embedding_model, s.enable_faiss)
        self.neo4j = Neo4jStore(s.neo4j_uri, s.neo4j_username, s.neo4j_password, s.neo4j_database, s.enable_neo4j)
        self.search = SearchService(self.sections, self.faiss)
        self.insights = InsightService(self.sections, self.neo4j)

container = AppContainer()
