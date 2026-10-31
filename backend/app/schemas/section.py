from pydantic import BaseModel, ConfigDict

class Section(BaseModel):
    model_config = ConfigDict(extra='ignore')
    section_number: str
    section_name: str
    chapter_number: str
    chapter_name: str
    chapter_subtype: str | None = None
    description: str

class SectionSummary(BaseModel):
    section_number: str
    section_name: str
    chapter_number: str
    chapter_name: str

class ChapterSummary(BaseModel):
    chapter_number: str
    chapter_name: str
    section_count: int
