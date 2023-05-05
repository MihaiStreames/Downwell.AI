import time
import pyautogui
import cv2 as cv
import numpy as np
import pygetwindow as gw

from imageProcessing import (
    extract_hp,
    killed_enemies,
    shop_side_room,
)

class CustomDownwellEnvironment:
    def __init__(self):
        self.game_window = None
        self.actions = ['left', 'right', 'space']

    def window_exists(self):
        self.game_window = gw.getWindowsWithTitle('Downwell')[1]

    def reset(self):
        if not self.game_window:
            self.window_exists()
        self.game_window.activate()

        time.sleep(0.5)

        pyautogui.press('esc')
        time.sleep(1)
        pyautogui.press('right')
        time.sleep(0.5)
        pyautogui.press('space')
        time.sleep(1)
        pyautogui.press('space')
        time.sleep(1)

        return self.capture_game_screen()

    def capture_game_screen(self):
        screenshot = pyautogui.screenshot(region=(self.game_window.left, self.game_window.top, self.game_window.width, self.game_window.height))
        game_screen = np.array(screenshot)

        return game_screen

    def step(self, action):
        action_key = self.actions[action]

        if action_key == 'space':
            pyautogui.keyDown('space')
            pyautogui.keyUp('space')
        else:
            pyautogui.keyDown(action_key)
            pyautogui.keyUp(action_key)

        next_state = self.capture_game_screen()
        reward, done = self.calculate(next_state)

        hp, killed, shop_side = self.game_info(next_state)

        if done:
            time.sleep(10)
            pyautogui.press('space')

        return next_state, reward, done

    def game_info(self, screen):
        hp = extract_hp(screen)
        killed = killed_enemies(screen)
        shop_side = shop_side_room(screen)

        return hp, killed, shop_side

    def calculate(self, screen):
        hp, killed_enemies, shop_side = self.game_info(screen)

        reward = 0
        done = False

        # Implement your reward calculation logic here based on the game state
        # ...

        # Implement the constraint to avoid entering shops or side rooms
        if shop_side:
            reward = -1000

        # Check if the game is over (HP reaches 0)
        if hp <= 0:
            done = True
            reward = -1000

        return reward, done