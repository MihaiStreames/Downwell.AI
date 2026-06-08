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


import ctypes
import ctypes.wintypes as wt
from dataclasses import dataclass

import cv2
import dxcam
from loguru import logger
import numpy as np
from src.consts import CROP_LEFT_RATIO
from src.consts import CROP_RIGHT_RATIO
from src.consts import IMAGE_HEIGHT
from src.consts import IMAGE_WIDTH
from src.consts import WINDOW_TITLE

from ._base import BaseCapture


_user32 = ctypes.windll.user32


class _Rect(ctypes.Structure):
    _fields_ = [("left", wt.LONG), ("top", wt.LONG), ("right", wt.LONG), ("bottom", wt.LONG)]


@dataclass
class _WindowRect:
    left: int
    top: int
    right: int
    bottom: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)

    def __getitem__(self, index: int) -> int:
        return [self.left, self.top, self.right, self.bottom][index]


@dataclass(frozen=True)
class _CapturedRegion:
    rect: _WindowRect
    image: np.ndarray


def _find_window_rect(title: str) -> _WindowRect | None:
    handle = _user32.FindWindowW(None, title)
    if not handle:
        logger.warning("failed to get handle")
        return None

    rect = _Rect()
    if not _user32.GetWindowRect(handle, ctypes.byref(rect)):
        logger.warning(f"failed to get window rect ({handle})")
        return None

    return _WindowRect(rect.left, rect.top, rect.right, rect.bottom)


class DXCamCapture(BaseCapture):
    """Capture and preprocess frames on Windows via ``dxcam``."""

    def __init__(self, title: str = WINDOW_TITLE) -> None:
        self.__title = title
        self.__camera = dxcam.create()  # if needed precise the buffer size (default is 8)

    def _as_region(self) -> _CapturedRegion | None:
        rect = _find_window_rect(self.__title)
        if rect is None:
            return None

        frame = self.__camera.grab(region=rect.as_tuple())
        if frame is None:
            logger.warning(f"failed to grab frame ({rect.as_tuple()})")
            return None

        return _CapturedRegion(rect, frame)

    def grab(self) -> np.ndarray | None:
        """Grab an image (grayscale), or None if window not found."""
        region = self._as_region()
        if region is None:
            return None

        width = region.rect[2] - region.rect[0]
        left = int(width * CROP_LEFT_RATIO)
        right = int(width * CROP_RIGHT_RATIO)
        gray = cv2.cvtColor(region.image[:, left:right], cv2.COLOR_BGR2GRAY)

        return cv2.resize(gray, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
