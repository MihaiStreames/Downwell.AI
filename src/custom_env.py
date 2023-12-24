import time
import pyautogui
import pygetwindow as gw


class CustomDownwellEnvironment:
    def __init__(self):
        self.gameWindow = None

        self.actions = {0: 'left', 1: 'right', 2: 'space'}

        self.multipliers = [10, 2, 3, -5]
        self.prevX = 0
        self.prevY = 0

    def window_exists(self):
        self.gameWindow = gw.getWindowsWithTitle('Downwell')[1]

    def get_game_window_dimensions(self):
        if self.gameWindow is None:
            self.window_exists()
        return self.gameWindow.left, self.gameWindow.top, self.gameWindow.width, self.gameWindow.height

    def get_state(self):
        raise NotImplementedError("get_state() not implemented")

    def is_game_over(self, player):
        return player.get_value('hp') == 0

    def reset(self, player):  # Whenever launching the bot, restart the game
        if not self.gameWindow:
            self.window_exists()
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

        if self.is_game_over(player):
            time.sleep(10)
            pyautogui.press('space')

    def calculate_reward(self, player):
        reward = 0

        if player.get_value('hp') > 0:
            reward += player.get_value('hp') * self.multipliers[0]
        if player.is_gem_high():
            reward += self.multipliers[1]
        reward += player.get_value('combo') * self.multipliers[2]

        if player.get_value('xpos') == self.prevX and player.get_value('ypos') == self.prevY:
            reward -= self.multipliers[3]
        else:
            self.prevX = player.get_value('xpos')
            self.prevY = player.get_value('ypos')

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

        nextState = self.get_state()
        reward = self.calculate_reward(player)

        done = self.is_game_over(player)
        return nextState, reward, done