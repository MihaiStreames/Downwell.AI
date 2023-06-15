import time
import cv2 as cv
import numpy as np
import pyautogui
import pygetwindow as gw

from imageProcessing import *

class CustomDownwellEnvironment:
    def __init__(self):
        self.game_window = None
        self.actions = ['left', 'right', 'space']
        self.prev_hp = None
        self.prev_gems = 0
        self.platform_counter = 0

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
        screenshot = pyautogui.screenshot(
            region=(self.game_window.left, self.game_window.top, self.game_window.width, self.game_window.height))
        game_screen = np.array(screenshot)

        cv.imshow('Game Screen', game_screen)
        cv.waitKey(1)

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

        if done:
            time.sleep(10)
            pyautogui.press('space')

        return next_state, reward, done

    def game_info(self, screen):
        hp, gems = extract_info(screen)
        return hp, gems

    def calculate(self, screen):
        hp, gems = self.game_info(screen)
        reward = 0
        done = False

        # Exploration bonus
        reward += 1

        # Reward for moving down
        if self.platform_counter > 10:
            reward -= 5

        # Reward for collecting gems
        if self.prev_gems is not None and gems > self.prev_gems:
            reward += 10 * (gems - self.prev_gems)
        self.prev_gems = gems

        # Reward for HP change
        if self.prev_hp is not None:
            hp_change = hp - self.prev_hp
            reward += hp_change * 10
        self.prev_hp = hp

        if hp <= 0:
            done = True
            reward = -1000

        return reward, done