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


if sys.platform != "linux":
    msg = "x11_capture is only supported on Linux"
    raise RuntimeError(msg)


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


def _find_window(display: Display, atom: int, title: str) -> tuple[Any, int, int] | None:
    root = display.screen().root

    prop = root.get_full_property(atom, X.AnyPropertyType)  # to avoid SIGSEGVs
    if prop is None:
        logger.warning(f"failed to get window list ({atom}, {root})")
        return None

    for wid in prop.value:
        win = display.create_resource_object("window", wid)

        try:
            name = win.get_wm_name()
            if not (name and title in name):
                continue

            geom = win.get_geometry()

        except (xlib_error.BadWindow, xlib_error.BadDrawable):
            logger.opt(exception=True).debug("failed to get window name, will skip")
            continue

        else:
            return win, geom.width, geom.height

    return None


class X11Capture(BaseCapture):
    """Capture and preprocess frames on Linux via ``X11``."""

    def __init__(self, title: str = WINDOW_TITLE) -> None:
        self.__title: str = title
        self.__display = Display()
        self.__atom = self.__display.intern_atom("_NET_CLIENT_LIST")

        self.__window: Any | None = None
        self.__width: int = 0
        self.__height: int = 0

    def grab(self) -> np.ndarray | None:
        """Grab an image (grayscale), or ``None`` if window not found."""
        if self.__window is None:
            result = _find_window(self.__display, self.__atom, self.__title)
            if result is None:
                return None

            self.__window, self.__width, self.__height = result

        try:
            raw = self.__window.get_image(0, 0, self.__width, self.__height, X.ZPixmap, 0xFFFFFFFF)
        except (xlib_error.BadDrawable, xlib_error.BadMatch):
            self.__window = None
            logger.opt(exception=True).warning("XGetImage failed, will re-locate next grab")
            return None

        frame = np.frombuffer(raw.data, dtype=np.uint8).reshape((self.__height, self.__width, 4))
        left = int(self.__width * CROP_LEFT_RATIO)
        right = int(self.__width * CROP_RIGHT_RATIO)
        gray = cv2.cvtColor(frame[:, left:right], cv2.COLOR_BGRA2GRAY)

        return cv2.resize(gray, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
