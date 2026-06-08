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

from src.consts import ACTION_KEYS


if sys.platform == "linux":
    from pynput.keyboard import Controller
    from pynput.keyboard import Key

    _KEY_MAP: dict[str, Key] = {
        "space": Key.space,
        "left": Key.left,
        "right": Key.right,
        "esc": Key.esc,
    }

if sys.platform == "win32":
    import pydirectinput


class InputHandler:
    """
    Cross-platform diff-based held-key input handler.

    Splits are based on the platform (``win32`` vs ``linux``), where ``win32`` uses ``pydirectinput`` and ``linux`` uses ``pynput``.
    Also provides a way to play sequences (in order) of keys via ``play_sequence(keys: list[str])``.
    """

    def __init__(self) -> None:
        self.__held: set[str] = set()

        if sys.platform == "linux":
            kb = Controller()

            def _tap(key: str) -> None:
                _press(key)
                _release(key)

            def _press(key: str) -> None:
                kb.press(_KEY_MAP[key])

            def _release(key: str) -> None:
                kb.release(_KEY_MAP[key])

            self.__tap = _tap
            self.__press = _press
            self.__release = _release

        if sys.platform == "win32":
            self.__tap = pydirectinput.press
            self.__press = pydirectinput.keyDown
            self.__release = pydirectinput.keyUp

    def apply(self, action: int) -> None:
        """Apply the given ``ACTION_KEYS`` int."""
        target = ACTION_KEYS[action]

        for key in self.__held - target:
            self.__release(key)
        for key in target - self.__held:
            self.__press(key)

        self.__held = target

    def release_all(self) -> None:
        """Release all held keys."""
        for key in self.__held:
            self.__release(key)
        self.__held.clear()

    def play_sequence(self, keys: list[str], delay: float = 0.5) -> None:
        """Play a sequence of keys in order, releasing all held keys first."""
        self.release_all()

        for key in keys:
            self.__tap(key)
            time.sleep(delay)
