from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.main import create_app
from app.repositories.csv_repository import CsvSectionRepository
from app.services.search_service import SearchService
from app.services.insight_service import InsightService

class Settings:
    app_version='test'; data_file=Path(__file__).parent/'fixtures'/'bns_sections.csv'

class FaissStub:
    available=False; error='not loaded'
    def load(self): pass
    def search(self, query, top_k): return []

class Neo4jStub:
    available=False; error='not connected'
    def connect(self): pass
    def close(self): pass
    def section_insights(self, section_number): return {'concepts':[],'cross_references':[],'related_sections':[]}

class TestContainer:
    def __init__(self):
        self.settings=Settings(); self.sections=CsvSectionRepository(self.settings.data_file); self.sections.load()
        self.faiss=FaissStub(); self.neo4j=Neo4jStub()
        self.search=SearchService(self.sections,self.faiss); self.insights=InsightService(self.sections,self.neo4j)

@pytest.fixture
def container(): return TestContainer()

@pytest.fixture
def client(container):
    with TestClient(create_app(container)) as c: yield c
