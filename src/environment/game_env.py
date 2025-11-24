from collections import deque
import time

import cv2
from loguru import logger
import numpy as np
from PIL import ImageGrab
import pyautogui
import pygetwindow as gw

from src.config import EnvConfig
from src.environment.capture import ScreenCapture


class CustomDownwellEnvironment:
    """Custom environment for Downwell game interaction.

    Parameters
    ----------
    config : EnvConfig
        Environment configuration.

    Attributes
    ----------
    game_window : pygetwindow.Win32Window | None
        Game window handle.
    image_size : tuple[int, int]
        Target image dimensions for processing.
    stack_size : int
        Number of frames to stack.
    actions : dict[int, set[str]]
        Mapping of action indices to key combinations.
    frame_stack : deque[np.ndarray]
        Rolling buffer of processed frames.
    capture_engine : ScreenCapture
        Screen capture utility.
    """

    def __init__(self, config: EnvConfig):
        self.game_window: pygetwindow.Win32Window = None
        self.image_size: tuple[int, int] = config.image_size
        self.stack_size: int = config.frame_stack
        self.actions: dict[int, set[str]] = {
            0: set(),
            1: {"space"},
            2: {"left"},
            3: {"right"},
            4: {"left", "space"},
            5: {"right", "space"},
        }
        self.frame_stack: deque[np.ndarray] = deque(maxlen=self.stack_size)

        self.capture_engine: ScreenCapture = ScreenCapture()
        self._capture_configured: bool = False

    def window_exists(self) -> bool:
        """Check if Downwell game window exists.

        Returns
        -------
        bool
            True if window found, False otherwise.
        """

        windows = gw.getWindowsWithTitle("Downwell")
        for window in windows:
            if window.title == "Downwell":
                self.game_window = window
                logger.info("Found Downwell window!")
                return True
        logger.warning("Downwell window not found.")
        self.game_window = None
        return False

    def get_game_window_dimensions(self) -> tuple[int, int, int, int]:
        """Get game window position and size.

        Returns
        -------
        tuple[int, int, int, int]
            Left, top, width, height of game window.

        Raises
        ------
        Exception
            If Downwell window cannot be found.
        """
        if self.game_window is None and not self.window_exists():
            raise Exception("Cannot find Downwell window!")
        return (
            self.game_window.left,
            self.game_window.top,
            self.game_window.width,
            self.game_window.height,
        )

    @staticmethod
    def crop_game_area(screenshot):
        """Crop screenshot to game play area.

        Parameters
        ----------
        screenshot : np.ndarray
            Full window screenshot.

        Returns
        -------
        np.ndarray
            Cropped game area (middle 40% horizontally).
        """
        height, width = screenshot.shape[:2]
        game_left = width * 3 // 10
        game_right = width * 7 // 10
        game_top = 0
        game_bottom = height
        return screenshot[game_top:game_bottom, game_left:game_right]

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for neural network input.

        Parameters
        ----------
        frame : np.ndarray
            Raw RGB/BGR frame.

        Returns
        -------
        np.ndarray
            Grayscale resized frame.
        """
        if frame is None:
            # Return a black frame if something goes wrong
            return np.zeros(self.image_size, dtype=np.uint8)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, self.image_size, interpolation=cv2.INTER_AREA)
        return resized_frame

    def get_state(self) -> np.ndarray | None:
        """Capture and process current game state.

        Returns
        -------
        np.ndarray | None
            Stacked frames as state, or None if capture fails or stack not full.
        """
        try:
            left, top, width, height = self.get_game_window_dimensions()

            # Configure capture region on first call or if window moved
            if not self._capture_configured:
                self.capture_engine.set_region(left, top, width, height)
                self._capture_configured = True

            frame = self.capture_engine.capture()

            cropped_frame = self.crop_game_area(frame)
            processed_frame = self._preprocess_frame(cropped_frame)
            self.frame_stack.append(processed_frame)

            # Ensure the stack is full before returning a state
            if len(self.frame_stack) < self.stack_size:
                return None

            state = np.stack(self.frame_stack, axis=2)
            return state
        except Exception as e:
            logger.error(f"Screenshot error: {e}")
            self._capture_configured = False
            return None

    @staticmethod
    def is_game_over(player) -> bool:
        """Check if game is over (player HP <= 0).

        Parameters
        ----------
        player : Player
            Player memory reader instance.

        Returns
        -------
        bool
            True if game is over, False otherwise.
        """
        hp = player.get_value("hp")
        return hp is not None and hp <= 0

    def reset(self, player) -> np.ndarray | None:
        """Reset the game environment.

        Parameters
        ----------
        player : Player
            Player memory reader instance.

        Returns
        -------
        np.ndarray | None
            Initial stacked state, or None if reset fails.
        """
        logger.info("Resetting game...")

        if not self.window_exists():
            return None

        try:
            self.frame_stack.clear()

            self.game_window.restore()
            self.game_window.activate()
            time.sleep(0.2)

            # Game reset sequence
            pyautogui.press("esc")
            time.sleep(0.2)
            pyautogui.press("right")
            time.sleep(0.2)
            pyautogui.press("space")
            time.sleep(0.2)
            pyautogui.press("space")
            time.sleep(2)

            # If game over, restart
            if self.is_game_over(player):
                logger.debug("Game over detected, restarting...")
                for _ in range(5):
                    pyautogui.press("space")
                    time.sleep(0.2)

            time.sleep(1)
            self.game_window.restore()
            self.game_window.activate()

            # Populate the frame stack with the initial frame to ensure it's full
            initial_screenshot = self.get_state()
            if initial_screenshot is None:
                # Manually grab one frame to start the process
                left, top, width, height = self.get_game_window_dimensions()

                screenshot = ImageGrab.grab(bbox=(left, top, left + width, top + height))
                frame = np.array(screenshot, dtype=np.uint8)
                if frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                cropped_frame = self.crop_game_area(frame)
                processed_frame = self._preprocess_frame(cropped_frame)
            else:
                processed_frame = initial_screenshot[:, :, -1]  # get last frame

            for _ in range(self.stack_size):
                self.frame_stack.append(processed_frame)

            return np.stack(self.frame_stack, axis=2)

        except Exception as e:
            logger.error(f"Error resetting game: {e}")
            return None
