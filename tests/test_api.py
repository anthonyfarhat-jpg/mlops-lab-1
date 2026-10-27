from fastapi.testclient import TestClient

from food11 import serve


class FakeModel:
    def predict(self, _input):
        return [[10.0] + [0.0] * 10]


def test_health(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    with TestClient(serve.app) as client:
        assert client.get("/health").status_code == 200


def test_predict_returns_valid_json(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    with TestClient(serve.app) as client:
        monkeypatch.setattr(serve, "model", FakeModel())
        with open("tests/fixtures/bread.jpg", "rb") as fixture:
            response = client.post(
                "/predict", files={"file": ("bread.jpg", fixture, "image/jpeg")}
            )
    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "Bread"
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_without_model_returns_503(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    monkeypatch.setattr(serve, "model", None)
    with TestClient(serve.app) as client:
        response = client.post(
            "/predict", files={"file": ("bread.jpg", b"not-used", "image/jpeg")}
        )
    assert response.status_code == 503
