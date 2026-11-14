from pydantic import BaseModel

from .section import SectionSummary


class RelatedSection(BaseModel):
    section: SectionSummary
    score: float | None = None
    relationship: str
    explanation: str | None = None


class SectionInsights(BaseModel):
    section_number: str
    concepts: list[str] = []
    cross_references: list[str] = []
    related_sections: list[RelatedSection] = []
    partial_ml: bool = False
    warnings: list[str] = []
