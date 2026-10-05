from fastapi.testclient import TestClient
from app.main import create_app

def test_health_ok_when_dependencies_ready():
    app = create_app(testing=True)
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
