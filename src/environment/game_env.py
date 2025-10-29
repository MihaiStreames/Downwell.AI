import time
from collections import deque
from typing import Tuple, Optional

import cv2
import numpy as np
import pyautogui
import pygetwindow as gw

from config import EnvConfig
from environment.capture import ScreenCapture


class CustomDownwellEnvironment:
    def __init__(self, config: EnvConfig):
        self.game_window = None
        self.image_size = config.image_size
        self.stack_size = config.frame_stack
        self.actions = {
            0: set(),
            1: {'space'},
            2: {'left'},
            3: {'right'},
            4: {'left', 'space'},
            5: {'right', 'space'}
        }
        self.frame_stack = deque(maxlen=self.stack_size)

        self.capture_engine = ScreenCapture()
        self._capture_configured = False

    def window_exists(self) -> bool:
        windows = gw.getWindowsWithTitle('Downwell')
        for window in windows:
            if window.title == 'Downwell':
                self.game_window = window
                print(f"Found Downwell window!")
                return True
        print("Downwell window not found.")
        self.game_window = None
        return False

    def get_game_window_dimensions(self) -> Tuple[int, int, int, int]:
        if self.game_window is None:
            if not self.window_exists():
                raise Exception("Cannot find Downwell window!")
        return self.game_window.left, self.game_window.top, self.game_window.width, self.game_window.height

    @staticmethod
    def crop_game_area(screenshot):
        height, width = screenshot.shape[:2]
        game_left = width * 3 // 10
        game_right = width * 7 // 10
        game_top = 0
        game_bottom = height
        return screenshot[game_top:game_bottom, game_left:game_right]

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            # Return a black frame if something goes wrong
            return np.zeros(self.image_size, dtype=np.uint8)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, self.image_size, interpolation=cv2.INTER_AREA)
        return resized_frame

    def get_state(self) -> Optional[np.ndarray]:
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
            print(f"Screenshot error: {e}")
            self._capture_configured = False
            return None

    @staticmethod
    def is_game_over(player) -> bool:
        hp = player.get_value('hp')
        return hp is not None and hp <= 0

    def reset(self, player) -> Optional[np.ndarray]:
        print("Resetting game...")

        if not self.window_exists():
            return None

        try:
            self.frame_stack.clear()

            self.game_window.restore()
            self.game_window.activate()
            time.sleep(0.2)

            # Game reset sequence
            pyautogui.press('esc')
            time.sleep(0.2)
            pyautogui.press('right')
            time.sleep(0.2)
            pyautogui.press('space')
            time.sleep(0.2)
            pyautogui.press('space')
            time.sleep(2)

            # If game over, restart
            if self.is_game_over(player):
                print("Game over detected, restarting...")
                for _ in range(5):
                    pyautogui.press('space')
                    time.sleep(0.2)

            time.sleep(1)
            self.game_window.restore()
            self.game_window.activate()

            # Populate the frame stack with the initial frame to ensure it's full
            initial_screenshot = self.get_state()
            if initial_screenshot is None:
                # Manually grab one frame to start the process
                left, top, width, height = self.get_game_window_dimensions()
                import PIL.ImageGrab as ImageGrab
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
            print(f"Error resetting game: {e}")
            return None
