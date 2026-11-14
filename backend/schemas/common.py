from pydantic import BaseModel


class Availability(BaseModel):
    source_data: bool
    keyword_search: bool
    semantic_search: bool
    graph_insights: bool


class HealthResponse(BaseModel):
    status: str
    version: str
    availability: Availability
