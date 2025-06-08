from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class GameState:
    """Immutable game state snapshot"""
    screenshot: Optional[np.ndarray]
    hp: float
    gems: float
    combo: float
    xpos: float
    ypos: float
    ammo: float
    gem_high: bool
    timestamp: float
    frame_id: int
