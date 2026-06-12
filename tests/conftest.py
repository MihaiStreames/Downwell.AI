import numpy as np
from src.consts import ACTION_BUFFER_DIM
from src.consts import FRAME_STACK
from src.consts import IMAGE_HEIGHT
from src.consts import IMAGE_WIDTH
from src.consts import RAM_DIM


_rng = np.random.default_rng()  # new way of handling random number generation


# from src.env.memory import MemoryState
# from src.consts import CENTER_XPOS


# def make_state(**kwargs) -> MemoryState:
#     defaults = {"ypos": 0.0, "xpos": CENTER_XPOS, "hp": 3, "gems": 0, "ammo": 3, "gem_high": 0, "combo": 0}
#     defaults.update(kwargs)
#     return MemoryState(**defaults)


def rand_img() -> np.ndarray:
    return _rng.integers(0, 256, (FRAME_STACK, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)


def rand_ram() -> np.ndarray:
    return _rng.random(RAM_DIM).astype(np.float32)


def rand_acts() -> np.ndarray:
    return _rng.random(ACTION_BUFFER_DIM).astype(np.float32)
