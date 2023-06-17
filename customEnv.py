import time
import pyautogui
import pygetwindow as gw
import numpy as np


class CustomDownwellEnvironment:
    def __init__(self):
        self.gameWindow = None

        self.actions = {0: 'left', 1: 'right', 2: 'space'}

        self.multipliers = [10, 2, 3, -5]
        self.prevX = 0
        self.prevY = 0

    def windowExists(self):
        self.gameWindow = gw.getWindowsWithTitle('Downwell')[1]

    def getGameWindowDimensions(self):
        if self.gameWindow is None:
            self.windowExists()
        return self.gameWindow.left, self.gameWindow.top, self.gameWindow.width, self.gameWindow.height

    def isGameOver(self, player):
        return player.getValue('hp') == 0

    def reset(self, player):  # Whenever launching the bot, restart the game
        if not self.gameWindow:
            self.windowExists()
        self.gameWindow.activate()

        time.sleep(0.5)

        pyautogui.press('esc')
        time.sleep(1)
        pyautogui.press('right')
        time.sleep(0.5)
        pyautogui.press('space')
        time.sleep(1)
        pyautogui.press('space')
        time.sleep(1)

        if self.isGameOver(player):
            time.sleep(10)
            pyautogui.press('space')

    def calculateReward(self, player):
        reward = 0

        if player.getValue('hp') > 0:
            reward += player.getValue('hp') * self.multipliers[0]
        if player.isGemHigh():
            reward += self.multipliers[1]
        reward += player.getValue('combo') * self.multipliers[2]

        if player.getValue('xpos') == self.prevX and player.getValue('ypos') == self.prevY:
            reward -= self.multipliers[3]
        else:
            self.prevX = player.getValue('xpos')
            self.prevY = player.getValue('ypos')

        return reward

    def step(self, action, player):
        action = self.actions[action]
        if action == 'left':
            pyautogui.press('left')
        elif action == 'right':
            pyautogui.press('right')
        elif action == 'space':
            pyautogui.press('space')
        else:
            raise ValueError(f"Unknown action: {action}")

        time.sleep(0.1)

        nextState = self.getState()
        reward = self.calculateReward(player)

        done = self.isGameOver(player)
        return nextState, reward, done