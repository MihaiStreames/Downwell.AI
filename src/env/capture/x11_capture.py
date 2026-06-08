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


from typing import Any

import cv2
from loguru import logger
import numpy as np
from src.consts import CROP_LEFT_RATIO
from src.consts import CROP_RIGHT_RATIO
from src.consts import IMAGE_HEIGHT
from src.consts import IMAGE_WIDTH
from src.consts import WINDOW_TITLE
from Xlib import X
from Xlib import error as xlib_error
from Xlib.display import Display

from ._base import BaseCapture


def _find_window(display: Display, title: str) -> tuple[Any, int, int] | None:
    root = display.screen().root
    atom = display.intern_atom("_NET_CLIENT_LIST")  # to avoid SIGSEGVs

    prop = root.get_full_property(atom, X.AnyPropertyType)
    if prop is None:
        logger.error(f"failed to get window list ({atom}, {root})")
        return None

    for wid in prop.value:
        win = display.create_resource_object("window", wid)

        try:
            name = win.get_wm_name()
            if not (name and title in name):
                continue

            geom = win.get_geometry()

        except xlib_error.XError:
            logger.opt(exception=True).warning("failed to get window name, will skip")
            continue

        else:
            return win, geom.width, geom.height

    return None


class X11Capture(BaseCapture):
    """Capture and preprocess frames on Linux via ``X11``."""

    def __init__(self, title: str = WINDOW_TITLE) -> None:
        self._title: str = title
        self._display = Display()
        self._window: Any | None = None
        self._width: int | None = None
        self._height: int | None = None

    def grab(self) -> np.ndarray | None:
        """Grab an image (grayscale), or None if window not found."""
        self._window, self._width, self._height = _find_window(self._display, self._title)
        if self._window is None:
            return None

        try:
            raw = self._window.get_image(0, 0, self._width, self._height, X.ZPixmap, 0xFFFFFFFF)
        except xlib_error.XError:
            self._window = None
            logger.opt(exception=True).warning("XGetImage failed, will re-locate next grab")
            return None

        frame = np.frombuffer(raw.data, dtype=np.uint8).reshape((self._height, self._width, 4))
        left = int(self._width * CROP_LEFT_RATIO)
        right = int(self._width * CROP_RIGHT_RATIO)
        gray = cv2.cvtColor(frame[:, left:right], cv2.COLOR_BGRA2GRAY)

        return cv2.resize(gray, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
