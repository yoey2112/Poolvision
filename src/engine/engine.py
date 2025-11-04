"""
Core orchestration engine for PoolVision.

This module glues together the vision subsystem (ball detection and tracking),
the rules subsystem (8‑ball and 9‑ball state machines), and the database to
provide high‑level operations such as applying a shot observation and
computing scores.

Currently, the engine contains stub implementations for demonstration
purposes only.  You will need to implement full logic for:

* Invoking the vision module to detect balls and assign IDs.
* Passing observations to the appropriate rules engine (eightball/nineball).
* Managing player turns, fouls, and shot histories.
* Storing results to a persistent database.
"""

from dataclasses import dataclass
from typing import Dict, Any

from ..vision.ball_tracking import detect_balls
from ..rules.eightball import EightBallGame
from ..rules.nineball import NineBallGame


@dataclass
class ShotObservation:
    """Represents a single observation (frame) of a shot."""
    frame_path: str
    timestamp: float


class Engine:
    """PoolVision engine orchestrating vision, rules, and persistence."""

    def __init__(self, game_type: str = "eightball") -> None:
        self.game_type = game_type
        self._init_game()

    def _init_game(self) -> None:
        if self.game_type == "eightball":
            self.game = EightBallGame()
        elif self.game_type == "nineball":
            self.game = NineBallGame()
        else:
            raise ValueError(f"Unsupported game type: {self.game_type}")

    def apply_shot(self, observation: ShotObservation) -> Dict[str, Any]:
        """
        Process a single shot observation.

        1. Run ball detection on the provided frame.
        2. Update the game state via the rules engine.
        3. Return a summary of the shot and updated scores.

        In this stub implementation, only ball detection is run and
        the game state is not modified.
        """
        # Run ball detection.  This returns a list of balls with IDs and positions.
        balls = detect_balls(observation.frame_path)

        # TODO: update game state based on detected balls and rules.
        # For now we simply return the detected balls and a placeholder score.
        return {
            "detected_balls": balls,
            "score": self.get_score(),
        }

    def get_score(self) -> Dict[str, int]:
        """Return the current score for the match."""
        # TODO: return actual scores from the rules engine.
        return {"player1": 0, "player2": 0}
