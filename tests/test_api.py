"""
Integration tests for FastAPI endpoints: /, /health, /feedback, /logs, and query validation.
"""
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "endpoints" in data
    assert data["service"] == "Finance Market Intelligence API"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "database_connected" in data


def test_feedback_endpoint():
    payload = {
        "session_id": "test-session-999",
        "query": "Test query for feedback",
        "rating": 1,
        "feedback_text": "Great insights on IT sector",
        "correction": "None",
    }
    response = client.post("/feedback", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "feedback_id" in data


def test_logs_endpoint():
    response = client.get("/logs?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "queries" in data
    assert "feedback" in data


def test_query_validation():
    # Query must be at least 2 chars
    response = client.post("/query", json={"session_id": "s1", "query": "a"})
    assert response.status_code == 422
