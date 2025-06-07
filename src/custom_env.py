import time
from typing import Tuple, Optional

import cv2
import numpy as np
import pyautogui
import pygetwindow as gw


class CustomDownwellEnvironment:
    def __init__(self):
        self.gameWindow = None
        self.actions = {
            0: 'none',
            1: 'left',
            2: 'right',
            3: 'space',
            4: 'left+space',
            5: 'right+space'
        }

        # Tracking variables
        self.prevX = None
        self.prevY = 0
        self.last_hp = 0
        self.last_gems = 0
        self.last_combo = 0
        self.steps_without_progress = 0
        self.last_significant_y = 0
        self.immobile_steps = 0

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

        self.gameWindow.activate()
        time.sleep(0.02)
        return self.gameWindow.left, self.gameWindow.top, self.gameWindow.width, self.gameWindow.height

    def get_state(self) -> Optional[np.ndarray]:
        left, top, width, height = self.get_game_window_dimensions()
        screenshot = pyautogui.screenshot(region=(left, top, width, height))

        # Convert to numpy array
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    @staticmethod
    def show_ai_vision(state, action_info=""):
        if state is not None:
            display_frame = state.copy()

            if action_info:
                cv2.putText(display_frame, action_info, (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            cv2.imshow('AI Vision', display_frame)
            cv2.waitKey(1)

    @staticmethod
    def is_game_over(player) -> bool:
        hp = player.get_value('hp')
        return hp is not None and hp <= 0

    def reset(self, player) -> Optional[np.ndarray]:  # Whenever launching the bot, restart the game
        print("Resetting game...")

        if not self.window_exists():
            return None

        try:
            self.gameWindow.restore()
            self.gameWindow.activate()
            time.sleep(0.2)

            # Reset tracking variables
            self.prevX = None
            self.prevY = 0
            self.last_hp = 0
            self.last_gems = 0
            self.last_combo = 0
            self.steps_without_progress = 0
            self.last_significant_y = 0
            self.immobile_steps = 0

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
                pyautogui.press('space')
                time.sleep(1)

            # Wait for the game to reset
            time.sleep(2)
            self.gameWindow.restore()
            self.gameWindow.activate()

            return self.get_state()

        except Exception as e:
            print(f"Error resetting game: {e}")
            return None

    def calculate_reward(self, player) -> float:
        current_hp = player.get_value('hp') or 0
        current_gems = player.get_value('gems') or 0
        current_combo = player.get_value('combo') or 0

        current_x = player.get_value('xpos')
        current_y = player.get_value('ypos') or 0

        reward = 0

        # Base survival reward
        if current_hp > 0:
            reward += 0.01

        # Health rewards
        if current_hp >= 4:
            reward += 0.5
        elif current_hp >= 2:
            reward += 0.2

        # Gem collection rewards
        gem_diff = current_gems - self.last_gems
        if gem_diff > 0:
            gem_reward = min(gem_diff * 0.2, 2.0)
            reward += gem_reward
            print(f"Collected {gem_diff} gems, reward: +{gem_reward:.2f}")

        # Combo system
        combo_diff = current_combo - self.last_combo
        if combo_diff > 0:
            combo_inc_reward = min(combo_diff * 1.0, 3.0)
            reward += combo_inc_reward
            print(f"Combo increased by {combo_diff}, reward: +{combo_inc_reward:.2f}")

        # Combo maintenance
        if current_combo > 0:
            combo_maintain_reward = min(current_combo * 0.05, 0.5)
            reward += combo_maintain_reward

        # Combo breaking penalty
        if self.last_combo > 0 and current_combo == 0:
            penalty = min(self.last_combo * 2, 10)
            reward -= penalty
            print(f"Combo broken! Lost {self.last_combo} combo, penalty: -{penalty:.1f}")

        # Gem high bonus
        if player.is_gem_high():
            reward += 25.0

        # Main objective: downward movement
        y_diff = self.prevY - current_y
        if y_diff > 5.0:
            scaled_reward = min(y_diff / 20.0, 2.0)
            reward += scaled_reward
            self.steps_without_progress = 0
            self.last_significant_y = current_y
            print(f"Downward progress: {y_diff:.1f} pixels, reward: +{scaled_reward:.2f}")
        else:
            self.steps_without_progress += 1

        if self.steps_without_progress > 10:
            reward -= 2.0
        if self.steps_without_progress > 20:
            reward -= 5.0

        # Handle immobility
        if current_x is not None and self.prevX is not None:
            # Check if truly immobile (both X and Y)
            if abs(current_x - self.prevX) < 0.1 and abs(current_y - self.prevY) < 0.1:
                self.immobile_steps += 1
                if self.immobile_steps > 5:
                    reward -= 1.0 * self.immobile_steps
            else:
                self.immobile_steps = 0
        else:
            # If X position unavailable, only check Y immobility
            if abs(current_y - self.prevY) < 0.1:
                self.immobile_steps += 1
                if self.immobile_steps > 8:
                    reward -= 0.5 * self.immobile_steps
            else:
                self.immobile_steps = 0

        # Damage penalty
        hp_diff = self.last_hp - current_hp
        if hp_diff > 0:
            reward -= hp_diff * 3

        # Update tracking variables
        if current_x is not None:
            self.prevX = current_x
        self.prevY = current_y
        self.last_hp = current_hp
        self.last_gems = current_gems
        self.last_combo = current_combo

        if abs(reward) > 5:
            print(
                f"High reward event: {reward:.2f} (HP:{current_hp}, Gems:{current_gems}, Combo:{current_combo}, Y-diff:{y_diff:.1f})")

        return reward

    def step(self, action_data, player) -> Tuple[Optional[np.ndarray], float, bool]:
        action_type, duration = action_data

        if action_type not in self.actions:
            print(f"Unknown action: {action_type}")
            return None, -10, True

        # Execute action
        self.gameWindow.activate()
        action_keys = self.actions[action_type]

        if action_keys == 'none':
            time.sleep(duration)
        elif '+' in action_keys:
            keys = action_keys.split('+')
            for key in keys:
                pyautogui.keyDown(key)
            time.sleep(duration)
            for key in keys:
                pyautogui.keyUp(key)
        else:
            pyautogui.keyDown(action_keys)
            time.sleep(duration)
            pyautogui.keyUp(action_keys)

        next_state = self.get_state()

        action_info = f"{self.actions[action_type]}({duration:.2f}s)"
        self.show_ai_vision(next_state, action_info)

        reward = self.calculate_reward(player)
        done = self.is_game_over(player)

        # Death penalty
        if done:
            reward -= 100
            print("Game over! Applying death penalty.")

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
