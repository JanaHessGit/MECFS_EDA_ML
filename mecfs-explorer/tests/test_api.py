"""
These tests exercise the app's structure (routes exist, validation works)
without requiring a trained model on disk, so `pytest` works right after
`git clone` — model-dependent tests are separated out and skipped if no
model artifact is present yet.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

MODEL_PRESENT = (
    Path(__file__).parent.parent / "data" / "processed" / "model" / "model.pt"
).exists()


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "disclaimer" in resp.json()


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_missing_body_is_rejected():
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


@pytest.mark.skipif(not MODEL_PRESENT, reason="No trained model artifact yet")
def test_predict_with_unknown_feature_rejected():
    resp = client.post("/predict", json={"features": {"not_a_real_feature": 1.0}})
    assert resp.status_code == 422


@pytest.mark.skipif(not MODEL_PRESENT, reason="No trained model artifact yet")
def test_model_metadata_includes_disclaimer():
    resp = client.get("/model/metadata")
    assert resp.status_code == 200
    assert "disclaimer" in resp.json()
