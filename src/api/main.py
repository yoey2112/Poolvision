"""
FastAPI application entry point for PoolVision.

This module exposes HTTP endpoints for interacting with the vision engine,
managing matches, updating player profiles, and retrieving scoreboard data.
Currently, only basic health‑check and stub endpoints are implemented.

To run the server:

    uvicorn src.api.main:app --reload

You can then access the automatic documentation at http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..engine.engine import Engine, ShotObservation

app = FastAPI(title="PoolVision API", version="0.1.0")

# Instantiate the core Engine once at startup.
engine = Engine()


class ObservationInput(BaseModel):
    """Schema for incoming shot observations from the vision module."""
    frame_path: str  # path to the image file for this observation
    timestamp: float


@app.get("/health", tags=["System"])
def health_check() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.post("/shot", tags=["Game"])
def register_shot(obs: ObservationInput) -> dict[str, str]:
    """
    Accepts a shot observation, processes it through the vision module,
    updates the game state, and returns a summary of the shot.

    For now this endpoint returns a placeholder response; you should
    implement full detection and rule integration in `Engine.apply_shot`.
    """
    try:
        shot_obs = ShotObservation(frame_path=obs.frame_path, timestamp=obs.timestamp)
        result = engine.apply_shot(shot_obs)
        return result
    except Exception as e:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/score", tags=["Game"])
def get_score() -> dict[str, int]:
    """
    Returns the current score for the ongoing match.  For 8‑ball, this
    would include the number of solids and stripes pocketed; for 9‑ball,
    it should return the APA point total.
    """
    return engine.get_score()
