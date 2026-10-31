from app.schemas.search import SearchRequest

def test_exact_section_lookup(container):
    result=container.search.search(SearchRequest(query='316'))
    assert result.mode_used=='exact'
    assert result.results[0].section.section_number=='316'

def test_keyword_fallback_when_faiss_unavailable(container):
    result=container.search.search(SearchRequest(query='entrusted property'))
    assert result.mode_used=='keyword'
    assert result.partial_ml is True
    assert result.results[0].section.section_number=='316'
