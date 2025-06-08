import queue
import threading
import time

import pyautogui

from models.action import Action


class ActorThread(threading.Thread):
    """Executes actions without blocking perception"""

    def __init__(self, env, action_queue):
        super().__init__(daemon=True)
        self.env = env
        self.action_queue = action_queue
        self.running = True

    def run(self):
        while self.running:
            try:
                try:
                    action = self.action_queue.get_nowait()
                    self.execute_action(action)
                except queue.Empty:
                    time.sleep(0.001)
            except Exception as e:
                print(f"Actor error: {e}")

    def execute_action(self, action: Action):
        action_info = self.env.actions[action.action_type]
        keys = action_info['keys']

        for key in keys:
            pyautogui.keyDown(key)
        time.sleep(action.duration)
        for key in keys:
            pyautogui.keyUp(key)

    def stop(self):
        self.running = False
