import sys


if sys.platform != "win32":
    raise NotImplementedError("Only Windows is supported.")

from collections import deque
import time
from typing import TYPE_CHECKING

import cv2
from loguru import logger
import numpy as np
from PIL import ImageGrab
import pyautogui
import pygetwindow as gw

from src.config import Config
from src.environment.capture import ScreenCapture
from src.utils.consts import ACTION_KEYS
from src.utils.consts import CROP_LEFT_RATIO
from src.utils.consts import CROP_RIGHT_RATIO
from src.utils.consts import WINDOW_TITLE


if TYPE_CHECKING:
    from src.environment.mem_extractor import Player


def _crop_game_area(screenshot: np.ndarray) -> np.ndarray:
    height, width = screenshot.shape[:2]
    game_left = int(width * CROP_LEFT_RATIO)
    game_right = int(width * CROP_RIGHT_RATIO)
    return screenshot[0:height, game_left:game_right]


def _is_game_over(player: "Player") -> bool:
    hp = player.get_value("hp")
    return hp is not None and hp <= 0


class CustomDownwellEnvironment:
    def __init__(self, config: Config) -> None:
        self._image_size: tuple[int, int] = config.image_size
        self._stack_size: int = config.frame_stack

        self.actions: dict[int, set[str]] = ACTION_KEYS

        self._capture_engine: ScreenCapture = ScreenCapture()
        self._frame_stack: deque[np.ndarray] = deque(maxlen=self._stack_size)

        self._game_window: gw.Win32Window | None = None
        self._capture_configured: bool = False

    def _window_exists(self) -> bool:
        windows = gw.getWindowsWithTitle(WINDOW_TITLE)
        for window in windows:
            if window.title == WINDOW_TITLE:
                self._game_window = window
                logger.info("Found Downwell window!")
                return True

        logger.warning("Downwell window not found.")

        self._game_window = None
        return False

    def _get_game_window_dimensions(self) -> tuple[int, int, int, int]:
        if self._game_window is None and not self._window_exists():
            raise Exception("Cannot find Downwell window!")
        assert self._game_window is not None
        return (
            self._game_window.left,
            self._game_window.top,
            self._game_window.width,
            self._game_window.height,
        )

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            return np.zeros(self._image_size, dtype=np.uint8)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, self._image_size, interpolation=cv2.INTER_AREA)
        return resized_frame

    def get_state(self) -> np.ndarray | None:
        try:
            left, top, width, height = self._get_game_window_dimensions()

            # configure capture region on first call or if window moved
            if not self._capture_configured:
                self._capture_engine.set_region(left, top, width, height)
                self._capture_configured = True

            frame = self._capture_engine.capture()

            cropped_frame = _crop_game_area(frame)
            processed_frame = self._preprocess_frame(cropped_frame)
            self._frame_stack.append(processed_frame)

            # ensure the stack is full before returning a state
            if len(self._frame_stack) < self._stack_size:
                return None

            state = np.stack(self._frame_stack, axis=2)
            return state

        except Exception as e:
            logger.error(f"Screenshot error: {e}")
            self._capture_configured = False
            return None

    def reset(self, player) -> np.ndarray | None:
        logger.info("Resetting game...")

        if not self._window_exists():
            return None

        try:
            self._frame_stack.clear()

            assert self._game_window is not None
            self._game_window.restore()
            self._game_window.activate()
            time.sleep(0.2)

            # game reset sequence
            pyautogui.press("esc")
            time.sleep(0.2)
            pyautogui.press("right")
            time.sleep(0.2)
            pyautogui.press("space")
            time.sleep(0.2)
            pyautogui.press("space")
            time.sleep(2)

            # if game over, restart
            if _is_game_over(player):
                logger.debug("Game over detected, restarting...")
                for _ in range(5):
                    pyautogui.press("space")
                    time.sleep(0.2)

            time.sleep(1)
            assert self._game_window is not None
            self._game_window.restore()
            self._game_window.activate()

            # populate the frame stack with the initial frame to ensure it's full
            initial_screenshot = self.get_state()
            if initial_screenshot is None:
                # manually grab one frame to start the process
                left, top, width, height = self._get_game_window_dimensions()

                screenshot = ImageGrab.grab(bbox=(left, top, left + width, top + height))
                frame = np.array(screenshot, dtype=np.uint8)
                if frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                cropped_frame = _crop_game_area(frame)
                processed_frame = self._preprocess_frame(cropped_frame)
            else:
                processed_frame = initial_screenshot[:, :, -1]  # get last frame

            for _ in range(self._stack_size):
                self._frame_stack.append(processed_frame)

            return np.stack(self._frame_stack, axis=2)

        except Exception as e:
            logger.error(f"Error resetting game: {e}")
            return None
