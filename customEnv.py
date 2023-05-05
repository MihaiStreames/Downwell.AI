import os
import time
import cv2 as cv
import numpy as np
import pyautogui
import pygetwindow as gw

from imageProcessing import extract_hp, find_player

# -------PATHS---------
dir = os.path.dirname(__file__)
standing_path = os.path.join(dir, 'templates', 'standing.png')
jumping_path = os.path.join(dir, 'templates', 'jumping.png')

standing = cv.imread(standing_path, 0)
jumping = cv.imread(jumping_path, 0)
# ---------------------

class CustomDownwellEnvironment:
    def __init__(self):
        self.game_window = None
        self.actions = ['left', 'right', 'space']
        self.prev_pos = None
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

        templates = [standing, jumping]  # Add more templates here
        top_left, bottom_right = find_player(game_screen, templates)

        if top_left and bottom_right:
            cv.rectangle(game_screen, top_left, bottom_right, (0, 255, 0), 2)
            player_pos = (top_left[1] + bottom_right[1]) // 2

            if self.prev_pos and 5 > abs(player_pos - self.prev_pos) > 0:
                self.platform_counter += 1
            else:
                self.platform_counter = 0

            self.prev_pos = player_pos

        cv.imshow('Game Screen with Hitboxes', game_screen)
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
        hp = extract_hp(screen)
        return hp

    def calculate(self, screen):
        hp = self.game_info(screen)
        reward = 0
        done = False

        # Exploration bonus
        reward += 1

        #enemies = detect_enemies(screen)
        #in_shop_bonus_area = detect_areas(screen)

        if self.platform_counter > 10:
            reward -= 5

        #if enemies:
        #    for enemy in enemies:
        #        distance = distance_enemy(screen, enemy)
        #        reward += distance

        reward += hp * 10

        #if in_shop_bonus_area:
        #    reward -= 1000

        if reward < -250:
            done = True

        if hp <= 0:
            done = True

        return reward, done