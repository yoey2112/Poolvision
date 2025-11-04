"""
SQLModel models for PoolVision.

The models defined here are used by the engine and API to store
players, matches, racks, and shots.  This is a minimal schema to get
started; you may need to extend it with additional fields (e.g.,
player skill levels for APA 8‑ball/9‑ball, match metadata, drill logs, etc.).
"""

from typing import Optional
from datetime import datetime
from sqlmodel import Field, SQLModel


class Player(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    apa_8ball_skill: Optional[int] = None
    apa_9ball_skill: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Match(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    player1_id: int = Field(foreign_key="player.id")
    player2_id: int = Field(foreign_key="player.id")
    game_type: str
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None


class Shot(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    match_id: int = Field(foreign_key="match.id")
    player_id: int = Field(foreign_key="player.id")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    description: Optional[str] = None  # e.g., JSON summarizing the shot
