from pathlib import Path
from typing import Any

import pytest

MODEL_PATH = Path("models/production/model.joblib")
METADATA_PATH = Path("models/production/metadata.json")


def client() -> Any:
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        pytest.skip("Production model artifacts are missing. Run `make pipeline`.")

    try:
        from fastapi.testclient import TestClient
    except RuntimeError as exc:
        pytest.skip(f"FastAPI TestClient unavailable: {exc}")

    from app.main import app

    return TestClient(app)


def valid_payload() -> dict[str, float]:
    return {
        "crim": 0.00632,
        "zn": 18.0,
        "indus": 2.31,
        "chas": 0.0,
        "nox": 0.538,
        "rm": 6.575,
        "age": 65.2,
        "dis": 4.09,
        "rad": 1.0,
        "tax": 296.0,
        "ptratio": 15.3,
        "b": 396.9,
        "lstat": 4.98,
    }


def test_health() -> None:
    response = client().get("/health")

    assert response.status_code == 200
    payload: dict[str, Any] = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["model_stage"] == "production"


def test_predict_valid_payload() -> None:
    response = client().post("/predict", json=valid_payload())

    assert response.status_code == 200
    payload: dict[str, Any] = response.json()
    assert isinstance(payload["prediction"], float)
    assert payload["model_stage"] == "production"
    assert payload["target"] == "medv"
    assert payload["features_used"] == list(valid_payload())


def test_metrics_endpoint() -> None:
    test_client = client()
    test_client.post("/predict", json=valid_payload())
    response = test_client.get("/metrics")

    assert response.status_code == 200
    body = response.text
    assert "prediction_requests_total" in body
    assert "prediction_latency_seconds" in body
    assert "model_loaded" in body
