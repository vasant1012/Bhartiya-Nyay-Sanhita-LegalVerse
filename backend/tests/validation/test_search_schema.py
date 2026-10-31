import pytest
from pydantic import ValidationError

from backend.schemas.search import SearchRequest


def test_blank_query_rejected():
    with pytest.raises(ValidationError): SearchRequest(query='   ')

def test_top_k_limit():
    with pytest.raises(ValidationError): SearchRequest(query='property', top_k=21)
