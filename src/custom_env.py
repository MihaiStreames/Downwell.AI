import time
from typing import Tuple, Optional

import cv2
import numpy as np
import pyautogui
import pygetwindow as gw


class CustomDownwellEnvironment:
    def __init__(self):
        self.gameWindow = None
        self.actions = {0: 'left', 1: 'right', 2: 'space'}
        self.multipliers = [10, 2, 3, -5]
        self.prevX = 0
        self.prevY = 0
        self.last_hp = 0
        self.last_gems = 0
        self.last_combo = 0
        self.frame_stack_size = 4
        self.frame_stack = []

        # Set pyautogui settings
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.01

    def window_exists(self) -> bool:
        windows = gw.getWindowsWithTitle('Downwell')
        if windows:
            self.gameWindow = windows[0]
            return True
        else:
            print("Downwell window not found! Make sure the game is running.")
            return False

    def get_game_window_dimensions(self) -> Tuple[int, int, int, int]:
        if self.gameWindow is None:
            if not self.window_exists():
                raise Exception("Cannot find Downwell window!")

        # Make sure window is active
        self.gameWindow.activate()
        time.sleep(0.1)
        return self.gameWindow.left, self.gameWindow.top, self.gameWindow.width, self.gameWindow.height

    def get_state(self) -> Optional[np.ndarray]:
        left, top, width, height = self.get_game_window_dimensions()
        screenshot = pyautogui.screenshot(region=(left, top, width, height))

        # Convert to numpy array
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to standard size
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)

        return frame

    @staticmethod
    def is_game_over(player) -> bool:
        return player.get_value('hp') <= 0

    def reset(self, player) -> Optional[np.ndarray]:  # Whenever launching the bot, restart the game
        print("Resetting game...")

        if not self.window_exists():
            return None

        try:
            self.gameWindow.activate()
            time.sleep(0.5)

            # Reset game state tracking
            self.prevX = 0
            self.prevY = 0
            self.last_hp = 0
            self.last_gems = 0
            self.last_combo = 0
            self.frame_stack = []

            # Game reset sequence
            pyautogui.press('esc')
            time.sleep(1)
            pyautogui.press('right')
            time.sleep(0.5)
            pyautogui.press('space')
            time.sleep(1)
            pyautogui.press('space')
            time.sleep(2)

            # If game over, restart
            if self.is_game_over(player):
                print("Game over detected, restarting...")
                time.sleep(1)
                pyautogui.press('space')
                time.sleep(2)

            # Wait for game to start
            time.sleep(1)

            return self.get_state()  # Get the initial state after reset

        except Exception as e:
            print(f"Error resetting game: {e}")
            return None

    def calculate_reward(self, player) -> float:
        current_hp = player.get_value('hp') or 0
        current_gems = player.get_value('gems') or 0
        current_combo = player.get_value('combo') or 0
        current_x = player.get_value('xpos') or 0
        current_y = player.get_value('ypos') or 0

        reward = 0

        # Base survival reward
        if current_hp > 0:
            reward += 1

        # Health-based reward
        reward += current_hp * self.multipliers[0]

        # Reward for gem collection
        gem_diff = current_gems - self.last_gems
        if gem_diff > 0:
            reward += gem_diff * 5

        # Reward for maintaining combo
        if current_combo > 0:
            reward += current_combo * self.multipliers[2]

        # Combo increase reward
        combo_diff = current_combo - self.last_combo
        if combo_diff > 0:
            reward += combo_diff * 10

        # Reward for gem high state
        if player.is_gem_high():
            reward += self.multipliers[1] * 50

        # Movement penalty (encourage exploration)
        if abs(current_x - self.prevX) < 0.1 and abs(current_y - self.prevY) < 0.1:
            reward += self.multipliers[3]  # Negative penalty

        # Downward movement reward (main objective)
        y_diff = self.prevY - current_y
        if y_diff > 0:  # Moving down
            reward += y_diff * 2

        # Update tracking variables
        self.prevX = current_x
        self.prevY = current_y
        self.last_hp = current_hp
        self.last_gems = current_gems
        self.last_combo = current_combo

        return reward

    def step(self, action: int, player) -> Tuple[Optional[np.ndarray], float, bool]:
        if action not in self.actions:
            print(f"Unknown action: {action}")
            return None, -10, True

        # Execute action
        pyautogui.press(self.actions[action])
        time.sleep(0.1)  # Wait for action to register

        # Get next state
        next_state = self.get_state()

        # Calculate reward
        reward = self.calculate_reward(player)

        # Check if game is over
        done = self.is_game_over(player)

        # Additional penalty for dying
        if done:
            reward -= 100

        return next_state, reward, done

    @staticmethod
    def get_debug_info(player) -> dict:
        return {
            'hp': player.get_value('hp'),
            'gems': player.get_value('gems'),
            'combo': player.get_value('combo'),
            'x': player.get_value('xpos'),
            'y': player.get_value('ypos'),
            'gem_high': player.is_gem_high()
        }
