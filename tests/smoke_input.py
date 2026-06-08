import time

import pytest
from src.consts import ACTION_RIGHT
from src.consts import ACTION_RIGHT_JUMP
from src.env.input import InputHandler


pytestmark = pytest.mark.smoke


def setup_module():
    print("\nFocus the game window, inputs fire in 2s...")
    time.sleep(2)


def test_apply_right_then_release():
    inp = InputHandler()
    inp.apply(ACTION_RIGHT)
    time.sleep(0.3)
    inp.release_all()


def test_apply_right_jump_then_release():
    inp = InputHandler()
    inp.apply(ACTION_RIGHT_JUMP)
    time.sleep(0.3)
    inp.release_all()
