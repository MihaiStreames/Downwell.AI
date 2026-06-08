import time

import numpy as np
import pytest
from src.consts import IMAGE_HEIGHT
from src.consts import IMAGE_WIDTH
from src.env.capture import Capture


pytestmark = pytest.mark.smoke


def setup_module():
    print("\nFocus the game window, capture fires in 2s...")
    time.sleep(2)


def test_grab_returns_correct_shape():
    frame = Capture().grab()
    assert frame is not None, "grab() returned None, is the game window open?"
    assert frame.shape == (IMAGE_HEIGHT, IMAGE_WIDTH), f"expected ({IMAGE_HEIGHT}, {IMAGE_WIDTH}), got {frame.shape}"
    assert frame.dtype == np.uint8
    print(f"  shape={frame.shape}  min={frame.min()}  max={frame.max()}")
