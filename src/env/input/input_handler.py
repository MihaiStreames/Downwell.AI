# Copyright 2023 MihaiStreames, UnderNowhere
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys
import time

import pydirectinput
from src.consts import ACTION_KEYS


if sys.platform == "linux":
    from pynput.keyboard import Controller


class InputHandler:
    """
    Cross-platform diff-based held-key input handler.

    Splits are based on the platform (``win32`` vs ``linux``), where ``win32`` uses ``pydirectinput`` and ``linux`` uses ``pynput``.
    Also provides a way to play sequences (in order) of keys via ``play_sequence(keys: list[str])``.
    """

    def __init__(self) -> None:
        self._held: set[str] = set()

        if sys.platform == "linux":
            kb = Controller()

            def _tap_linux(key: str) -> None:
                kb.press(key)
                kb.release(key)

            self._tap = _tap_linux
            self._press = kb.press
            self._release = kb.release

        if sys.platform == "win32":
            self._tap = pydirectinput.press
            self._press = pydirectinput.keyDown
            self._release = pydirectinput.keyUp

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
