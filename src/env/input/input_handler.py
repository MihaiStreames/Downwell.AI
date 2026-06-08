import sys
import time

import pydirectinput
from src.consts import ACTION_KEYS


if sys.platform != "win32":
    from pynput.keyboard import Controller


class InputHandler:
    """
    Cross-platform diff-based held-key input handler.

    Splits are based on the platform (``win32`` vs ``linux``), where ``win32`` uses ``pydirectinput`` and ``linux`` uses ``pynput``.
    Also provides a way to play sequences (in order) of keys via ``play_sequence(keys: list[str])``.
    """

    def __init__(self) -> None:
        if sys.platform == "linux":
            self._kb = Controller()
        self._held: set[str] = set()

    def _tap(self, key: str) -> None:
        if sys.platform == "linux":
            self._kb.press(key)
            self._kb.release(key)
        if sys.platform == "win32":
            pydirectinput.press(key)

    def _press(self, key: str) -> None:
        if sys.platform == "linux":
            self._kb.press(key)
        if sys.platform == "win32":
            pydirectinput.keyDown(key)

    def _release(self, key: str) -> None:
        if sys.platform == "linux":
            self._kb.release(key)
        if sys.platform == "win32":
            pydirectinput.keyUp(key)

    def apply(self, action: int) -> None:
        """Apply the given ``ACTION_KEYS`` int."""
        target = ACTION_KEYS[action]

        for key in self._held - target:
            self._release(key)
        for key in target - self._held:
            self._press(key)

        self._held = target

    def release_all(self) -> None:
        """Release all held keys."""
        for key in self._held:
            self._release(key)
        self._held.clear()

    def play_sequence(self, keys: list[str]) -> None:
        """
        Play a sequence of keys in order.

        Note:
        -----
            This method releases all held keys before playing the sequence.
            (unsure if it is necessary)
        """
        self.release_all()  # unsure if this is necessary

        for key in keys:
            self._tap(key)
            time.sleep(0.5)
