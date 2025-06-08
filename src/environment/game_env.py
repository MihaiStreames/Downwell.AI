import time
from typing import Tuple, Optional

import cv2
import numpy as np
import pyautogui
import pygetwindow as gw


class CustomDownwellEnvironment:
    def __init__(self):
        self.gameWindow = None
        # Each action is a 3-bit binary number
        # Bit 0: Left, Bit 1: Right, Bit 2: Space
        self.actions = {
            0: {'keys': [], 'name': 'none'},  # 000
            1: {'keys': ['space'], 'name': 'space'},  # 001
            2: {'keys': ['right'], 'name': 'right'},  # 010
            3: {'keys': ['right', 'space'], 'name': 'right+space'},  # 011
            4: {'keys': ['left'], 'name': 'left'},  # 100
            5: {'keys': ['left', 'space'], 'name': 'left+space'},  # 101
            6: {'keys': ['left', 'right'], 'name': 'left+right'},  # 110 (rare)
            7: {'keys': ['left', 'right', 'space'], 'name': 'all'}  # 111 (rare)
        }

        # Tracking variables
        self.previous_frame = None

    def window_exists(self) -> bool:
        windows = gw.getWindowsWithTitle('Downwell')
        for window in windows:
            if window.title == 'Downwell':
                self.gameWindow = window
                print(f"Found Downwell window!")
                return True
        print("Downwell window not found.")
        self.gameWindow = None
        return False

    def get_game_window_dimensions(self) -> Tuple[int, int, int, int]:
        if self.gameWindow is None:
            if not self.window_exists():
                raise Exception("Cannot find Downwell window!")
        return self.gameWindow.left, self.gameWindow.top, self.gameWindow.width, self.gameWindow.height

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

            # Handle different color formats
            if frame.shape[2] == 4:  # RGBA
                frame = frame[:, :, :3]
            elif frame.shape[2] == 3:  # Already RGB
                pass

            game_frame = self.crop_game_area(frame)

            # Create 6-channel state (RGB + movement difference)
            if self.previous_frame is not None:
                diff = cv2.absdiff(game_frame, self.previous_frame)
                state = np.concatenate([game_frame, diff], axis=2)
                self.previous_frame = game_frame.copy()
            else:
                self.previous_frame = game_frame.copy()
                return game_frame  # First frame, return RGB only

            return state
        except Exception as e:
            print(f"Screenshot error: {e}")
            return None

    @staticmethod
    def show_ai_vision(state, action_info="", player=None, q_values=None):
        if state is None:
            return

        if state.shape[2] >= 3:
            display_frame = state[:, :, :3].copy()
        else:
            display_frame = state.copy()

        h, w = display_frame.shape[:2]

        # Show action info (now includes HP, Gems, Combo, Ammo, Gem High)
        cv2.putText(display_frame, action_info, (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show only position info
        if player is not None:
            memory_info = [
                f"Y: {player.get_value('ypos') or 0:.0f}",
                f"X: {player.get_value('xpos') or 0:.0f}"
            ]

            for i, info in enumerate(memory_info):
                cv2.putText(display_frame, info, (5, 60 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Show Q-values if available
        if q_values is not None and len(q_values) > 0:
            action_names = ['none', 'space', 'right', 'right+space', 'left', 'left+space', 'left+right', 'all']

            if not all(q == 0 for q in q_values):
                max_q = max(q_values)
                min_q = min(q_values)
                q_range = max_q - min_q if max_q != min_q else 1

                # Show Q-value bars
                for i, (name, q_val) in enumerate(zip(action_names[:len(q_values)], q_values)):
                    if q_range > 0:
                        normalized_q = (q_val - min_q) / q_range
                    else:
                        normalized_q = 0.5

                    color = (0, 255, 0) if i == np.argmax(q_values) else (100, 100, 100)
                    bar_width = int(normalized_q * 100)

                    y_pos = 25 + i * 16
                    cv2.rectangle(display_frame, (w - 130, y_pos),
                                  (w - 130 + bar_width, y_pos + 10), color, -1)
                    cv2.putText(display_frame, f"{name}: {q_val:.1f}",
                                (w - 125, y_pos + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            else:
                cv2.putText(display_frame, "EXPLORING", (w - 130, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # Display window
        cv2.namedWindow('AI Vision', cv2.WINDOW_NORMAL)
        cv2.moveWindow('AI Vision', 100, 100)
        cv2.resizeWindow('AI Vision', display_frame.shape[1], display_frame.shape[0])
        cv2.imshow('AI Vision', display_frame)
        cv2.waitKey(1)

    @staticmethod
    def is_game_over(player) -> bool:
        hp = player.get_value('hp')
        return hp is not None and hp <= 0

    def reset(self, player) -> Optional[np.ndarray]:
        print("Resetting game...")

        if not self.window_exists():
            return None

        try:
            self.gameWindow.restore()
            self.gameWindow.activate()
            time.sleep(0.1)

            # Reset tracking variables
            self.previous_frame = None

            # Game reset sequence
            pyautogui.press('esc')
            time.sleep(0.2)
            pyautogui.press('right')
            time.sleep(0.1)
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
            self.gameWindow.restore()
            self.gameWindow.activate()

            return self.get_state()

        except Exception as e:
            print(f"Error resetting game: {e}")
            return None
