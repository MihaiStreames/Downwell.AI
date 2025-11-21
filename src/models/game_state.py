from dataclasses import dataclass

import numpy as np


@dataclass
class GameState:
    """Immutable game state snapshot"""

    screenshot: np.ndarray | None
    hp: float | None
    gems: float
    combo: float
    xpos: float | None
    ypos: float | None
    ammo: float
    gem_high: bool
    timestamp: float
    frame_id: int
