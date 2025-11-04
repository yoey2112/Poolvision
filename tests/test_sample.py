"""
Sample unit test for the PoolVision skeleton.

This test verifies that the FastAPI health check endpoint works and
that the engine returns a placeholder score when a shot is applied.
"""

import tempfile
import numpy as np  # type: ignore
import cv2  # type: ignore
from fastapi.testclient import TestClient

from src.api.main import app


def test_health_check() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_register_shot_returns_score() -> None:
    # Create a dummy black image to simulate a frame
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        cv2.imwrite(tmp.name, img)
        client = TestClient(app)
        payload = {"frame_path": tmp.name, "timestamp": 0.0}
        response = client.post("/shot", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "detected_balls" in data
        assert "score" in data