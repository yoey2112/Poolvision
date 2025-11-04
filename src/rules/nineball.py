"""
APA 9‑Ball rules implementation.

In APA 9‑ball, each ball pocketed is worth one point, except the 9‑ball
which is worth two points if pocketed early (depending on league rules).
This module provides a basic skeleton for tracking scores and switching
turns.  Full foul and turn logic must be implemented.
"""

from typing import Dict, Any


class NineBallGame:
    """Represents a 9‑ball game with APA scoring."""

    def __init__(self) -> None:
        # In APA 9‑ball, players accumulate points equal to the number of
        # balls pocketed (1–9).  The 9‑ball counts as two points if pocketed
        # on a break or combo before being legally dead.
        self.player_scores = {"player1": 0, "player2": 0}
        self.current_player = "player1"
        self.game_over = False

    def apply_shot(self, balls: list[dict[str, Any]]) -> Dict[str, Any]:
        """
        Update game state based on detected balls.  This stub simply toggles
        the current player and does not award points.  You should implement
        APA 9‑ball scoring (1 point per ball, 2 for the 9‑ball under
        specific conditions), foul detection, and ball‑in‑hand logic.
        """
        if self.game_over:
            return {"message": "Game already over"}

        # TODO: implement full 9‑ball scoring and rules.
        self.current_player = "player2" if self.current_player == "player1" else "player1"
        return {"message": "Turn completed", "next_player": self.current_player}

    def get_score(self) -> Dict[str, int]:
        """Return the current APA point total for each player."""
        return self.player_scores
