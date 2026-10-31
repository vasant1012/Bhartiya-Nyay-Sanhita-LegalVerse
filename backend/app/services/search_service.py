from __future__ import annotations
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.schemas.search import MatchEvidence, SearchRequest, SearchResponse, SearchResult
from app.schemas.section import SectionSummary

class SearchService:
    def __init__(self, repository, faiss_store):
        self.repository = repository
        self.faiss_store = faiss_store

    @staticmethod
    def _label(score: float) -> str:
        if score >= 0.75: return 'very_strong'
        if score >= 0.50: return 'strong'
        if score >= 0.25: return 'possible'
        return 'related'

    @staticmethod
    def _excerpt(text: str, limit: int = 320) -> str:
        text = ' '.join(text.split())
        return text if len(text) <= limit else text[:limit].rsplit(' ', 1)[0] + '...'

    def _keyword(self, query: str, top_k: int, chapter: str | None) -> list[dict]:
        sections = self.repository.list_sections(chapter)
        if not sections:
            return []
        documents = [f'{s.section_number} {s.section_name} {s.chapter_name} {s.description}' for s in sections]
        matrix = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True).fit_transform(documents + [query])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        order = scores.argsort()[::-1][:top_k]
        return [{'section': sections[i], 'score': float(scores[i])} for i in order if scores[i] > 0]

    def search(self, request: SearchRequest) -> SearchResponse:
        warnings, partial_ml = [], False
        exact = self.repository.get_section(request.query)
        if exact:
            candidates = [{'section': exact, 'score': 1.0, 'method': 'exact'}]
            mode_used = 'exact'
        else:
            keyword = self._keyword(request.query, request.top_k * 2, request.chapter_number)
            semantic = []
            if request.mode.value in {'recommended', 'semantic'}:
                semantic = self.faiss_store.search(request.query, request.top_k * 2)
                if not self.faiss_store.available:
                    partial_ml = True
                    warnings.append('Semantic search unavailable; keyword results used')
            if request.mode.value == 'semantic' and semantic:
                by_number = {str(s.section_number): s for s in self.repository.list_sections(request.chapter_number)}
                candidates = [{'section': by_number[str(x['section_number'])], 'score': x['score'], 'method': 'semantic'}
                              for x in semantic if str(x.get('section_number')) in by_number]
                mode_used = 'semantic'
            elif request.mode.value == 'recommended' and semantic:
                fused = {}
                for x in keyword:
                    fused[str(x['section'].section_number)] = {'section': x['section'], 'score': .4 * x['score'], 'method': 'hybrid'}
                by_number = {str(s.section_number): s for s in self.repository.list_sections(request.chapter_number)}
                for x in semantic:
                    key = str(x.get('section_number'))
                    if key in by_number:
                        fused.setdefault(key, {'section': by_number[key], 'score': 0.0, 'method': 'hybrid'})
                        fused[key]['score'] += .6 * max(0.0, min(1.0, float(x['score'])))
                candidates = sorted(fused.values(), key=lambda x: x['score'], reverse=True)
                mode_used = 'recommended'
            else:
                candidates = [{**x, 'method': 'keyword'} for x in keyword]
                mode_used = 'keyword'

        query_terms = set(re.findall(r'[a-zA-Z]{3,}', request.query.lower()))
        results = []
        for item in candidates[:request.top_k]:
            section = item['section']
            text_terms = set(re.findall(r'[a-zA-Z]{3,}', f'{section.section_name} {section.description}'.lower()))
            results.append(SearchResult(
                section=SectionSummary(**section.model_dump()),
                excerpt=self._excerpt(section.description),
                score=round(float(item['score']), 6),
                relevance=self._label(float(item['score'])),
                evidence=MatchEvidence(matched_terms=sorted(query_terms & text_terms)[:10], method=item['method'])
            ))
        return SearchResponse(query=request.query, mode_used=mode_used, results=results,
                              partial_ml=partial_ml, warnings=warnings)
