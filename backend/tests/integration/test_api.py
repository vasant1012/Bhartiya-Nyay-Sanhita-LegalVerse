def test_health(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["availability"]["source_data"] is True
    assert body["availability"]["semantic_search"] is False


def test_sections(client):
    response = client.get("/api/v1/sections")
    assert response.status_code == 200
    assert len(response.json()) == 3


def test_section_not_found(client):
    assert client.get("/api/v1/sections/999").status_code == 404


def test_search(client):
    response = client.post(
        "/api/v1/search",
        json={
            "query": "dishonestly entrusted property",
            "mode": "recommended",
            "top_k": 3,
        },
    )
    assert response.status_code == 200
    assert response.json()["results"][0]["section"]["section_number"] == "316"


def test_insight_fallback(client):
    response = client.get("/api/v1/insights/sections/316")
    assert response.status_code == 200
    assert response.json()["partial_ml"] is True
