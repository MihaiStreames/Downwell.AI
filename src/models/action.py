from dataclasses import dataclass


@dataclass
class Action:
    """Action to be executed"""
    action_type: int
    frame_id: int
