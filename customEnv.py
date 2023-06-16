import time
import pyautogui
import pygetwindow as gw

class CustomDownwellEnvironment:
    def __init__(self):
        self.game_window = None
        self.actions = ['left', 'right', 'space']

    def window_exists(self):
        self.game_window = gw.getWindowsWithTitle('Downwell')[1]

    def reset(self): # Whenever launching the game, reset the game
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