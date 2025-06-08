from dataclasses import dataclass


@dataclass
class Action:
    """Action to be executed"""
    action_type: int
    duration: float
    frame_id: int
