"""
APA 8‑Ball rules implementation.

This module defines a simple structure for managing an 8‑ball game.  The
current implementation does not yet enforce all rules; it serves as a
foundation to be extended with foul detection, turn changes, solids/stripes
assignment, and game completion logic.
"""

from typing import Dict, Any


class EightBallGame:
    """Represents an 8‑ball game with APA scoring."""

    def __init__(self) -> None:
        # Player scores (number of balls pocketed).  In APA 8‑ball, the
        # objective is to legally pocket your group (solids/stripes) and then
        # pocket the eight ball.
        self.player_scores = {"player1": 0, "player2": 0}
        self.current_player = "player1"
        self.game_over = False

    def apply_shot(self, balls: list[dict[str, Any]]) -> Dict[str, Any]:
        """
        Update game state based on detected balls.  This stub simply toggles
        the current player and does not score the shot.  You should
        implement: first contact ball determination, foul detection (e.g.,
        scratching, wrong ball), assignment of solids vs stripes, and
        completion logic for the eight ball.
        """
        if self.game_over:
            return {"message": "Game already over"}

        # TODO: implement full 8‑ball rules.
        # For now, alternate turns.
        self.current_player = "player2" if self.current_player == "player1" else "player1"
        return {"message": "Turn completed", "next_player": self.current_player}

    def get_score(self) -> Dict[str, int]:
        """Return the current score (balls pocketed) for each player."""
        return self.player_scores
