import time
from typing import Tuple, Optional

import cv2
import numpy as np
import pyautogui
import pygetwindow as gw

from config import EnvConfig


class CustomDownwellEnvironment:
    def __init__(self, config: EnvConfig):
        self.game_window = None
        self.image_size = config.image_size
        self.actions = {
            0: set(),
            1: {'space'},
            2: {'right'},
            3: {'left'},
        }
        self.previous_frame = None

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

    def get_state(self) -> Optional[np.ndarray]:
        try:
            left, top, width, height = self.get_game_window_dimensions()

            import PIL.ImageGrab as ImageGrab
            screenshot = ImageGrab.grab(bbox=(left, top, left + width, top + height))

            # Convert to numpy array
            frame = np.array(screenshot, dtype=np.uint8)
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]

            cropped_frame = self.crop_game_area(frame)

            # Create 6-channel state (RGB + movement difference)
            if self.previous_frame is not None:
                game_frame = cv2.resize(cropped_frame, self.image_size, interpolation=cv2.INTER_AREA)
                prev_frame = cv2.resize(self.previous_frame, self.image_size, interpolation=cv2.INTER_AREA)
                diff = cv2.absdiff(game_frame, prev_frame)
                state = np.concatenate([game_frame, diff], axis=2)
            else:
                game_frame = cv2.resize(cropped_frame, self.image_size, interpolation=cv2.INTER_AREA)
                state = np.concatenate([game_frame, game_frame], axis=2)

            self.previous_frame = cropped_frame.copy()

            return state

        except Exception as e:
            print(f"Screenshot error: {e}")
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
            self.game_window.restore()
            self.game_window.activate()
            time.sleep(0.2)

            # Reset tracking variables
            self.previous_frame = None

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

            return self.get_state()

        except Exception as e:
            print(f"Error resetting game: {e}")
            return None
