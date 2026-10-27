"""
Integration tests for Flask API Blueprint endpoints.
"""

import pytest
from flask import Flask
from src.api.routes import api_bp


@pytest.fixture
def test_client(temp_db):
    import src.api.routes as routes
    from src.persistence.repository import DocumentRepository, ChunkRepository
    routes._db = temp_db
    routes._doc_repo = DocumentRepository(temp_db)
    routes._chunk_repo = ChunkRepository(temp_db)

    app = Flask(__name__)
    app.register_blueprint(api_bp)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(test_client):
    res = test_client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_list_pdfs_endpoint(test_client):
    res = test_client.get("/api/pdfs")
    assert res.status_code == 200
    assert isinstance(res.get_json(), list)


def test_list_chunks_endpoint(test_client):
    res = test_client.get("/api/chunks?limit=10")
    assert res.status_code == 200
    data = res.get_json()
    assert "total" in data
    assert "chunks" in data
