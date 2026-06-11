from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health_returns_200() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200


def test_health_payload() -> None:
    data = client.get("/api/health").json()
    assert data["status"] == "ok"
    assert "version" in data


def test_ws_echo() -> None:
    with client.websocket_connect("/ws") as ws:
        ws.send_text("hello")
        msg = ws.receive_json()
        assert msg == {"echo": "hello"}
