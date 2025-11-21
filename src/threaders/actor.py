import queue
import threading
import time

from loguru import logger
import pyautogui


class ActorThread(threading.Thread):
    """Executes actions without blocking perception"""

    def __init__(self, env, action_queue):
        super().__init__(daemon=True)
        self.env = env
        self.action_queue = action_queue
        self.running = True
        self.currently_pressed = set()

    def run(self):
        while self.running:
            try:
                # Get the latest desired action from the Thinker
                action_cmd = self.action_queue.get_nowait()
                desired_keys = self.env.actions[action_cmd.action_type]

                # 1. Find keys that need to be released
                keys_to_release = self.currently_pressed - desired_keys
                for key in keys_to_release:
                    pyautogui.keyUp(key)

                # 2. Find keys that need to be pressed
                keys_to_press = desired_keys - self.currently_pressed
                for key in keys_to_press:
                    pyautogui.keyDown(key)

                # 3. Update the state of our currently pressed keys
                self.currently_pressed = desired_keys

            except queue.Empty:
                time.sleep(0.001)  # Sleep briefly if no new action
            except Exception as e:
                logger.error(f"Actor error: {e}")

    def stop(self):
        # When stopping, release all keys that are currently pressed
        for key in self.currently_pressed:
            pyautogui.keyUp(key)
        self.running = False
