from enum import Enum

from pydantic import BaseModel, Field, field_validator

from .section import SectionSummary


class SearchMode(str, Enum):
    recommended = 'recommended'
    keyword = 'keyword'
    semantic = 'semantic'


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    mode: SearchMode = SearchMode.recommended
    top_k: int = Field(default=5, ge=1, le=20)
    chapter_number: str | None = None

    @field_validator('query')
    @classmethod
    def clean_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('query must not be blank')
        return value


class MatchEvidence(BaseModel):
    matched_terms: list[str] = []
    concepts: list[str] = []
    method: str


class SearchResult(BaseModel):
    section: SectionSummary
    excerpt: str
    score: float
    relevance: str
    evidence: MatchEvidence


class SearchResponse(BaseModel):
    query: str
    mode_used: str
    results: list[SearchResult]
    partial_ml: bool = False
    warnings: list[str] = []
