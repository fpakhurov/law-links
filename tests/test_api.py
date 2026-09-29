from fastapi.testclient import TestClient

from main import app


def test_detect_and_health():
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "healthy"}
        response = client.post("/detect", json={"text": "пп. 1, 2 п. 2 ст. 3 НК РФ"})
        assert response.status_code == 200
        assert response.json() == {
            "links": [
                {"law_id": 15, "article": "3", "point_article": "2", "subpoint_article": "1"},
                {"law_id": 15, "article": "3", "point_article": "2", "subpoint_article": "2"},
            ]
        }


def test_detect_empty_text():
    with TestClient(app) as client:
        assert client.post("/detect", json={"text": ""}).json() == {"links": []}


def test_detect_validation_error():
    with TestClient(app) as client:
        assert client.post("/detect", json={}).status_code == 422
