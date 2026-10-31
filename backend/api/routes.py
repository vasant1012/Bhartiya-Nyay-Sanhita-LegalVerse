import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from fastapi import APIRouter, HTTPException, Query, Request

from backend.schemas.common import Availability, HealthResponse
from backend.schemas.insight import SectionInsights
from backend.schemas.search import SearchRequest, SearchResponse
from backend.schemas.section import ChapterSummary, Section, SectionSummary

router = APIRouter()

def c(request: Request):
    return request.app.state.container

@router.get('/health', response_model=HealthResponse, tags=['system'])
def health(request: Request):
    x = c(request)
    source = x.settings.data_file.exists()
    return HealthResponse(status='ok' if source else 'degraded', version=x.settings.app_version,
        availability=Availability(source_data=source, keyword_search=source,
                                  semantic_search=x.faiss.available, graph_insights=x.neo4j.available))

@router.get('/sections', response_model=list[SectionSummary], tags=['sections'])
def sections(request: Request, chapter_number: str | None = Query(default=None)):
    try:
        return [SectionSummary(**s.model_dump()) for s in c(request).sections.list_sections(chapter_number)]
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc))

@router.get('/sections/{section_number}', response_model=Section, tags=['sections'])
def section(section_number: str, request: Request):
    try:
        result = c(request).sections.get_section(section_number)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    if not result:
        raise HTTPException(status_code=404, detail='Section not found')
    return result

@router.get('/chapters', response_model=list[ChapterSummary], tags=['chapters'])
def chapters(request: Request):
    try:
        return c(request).sections.list_chapters()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc))

@router.post('/search', response_model=SearchResponse, tags=['search'])
def search(payload: SearchRequest, request: Request):
    try:
        return c(request).search.search(payload)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc))

@router.get('/insights/sections/{section_number}', response_model=SectionInsights, tags=['insights'])
def section_insights(section_number: str, request: Request):
    try:
        return c(request).insights.for_section(section_number)
    except KeyError:
        raise HTTPException(status_code=404, detail='Section not found')
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc))
