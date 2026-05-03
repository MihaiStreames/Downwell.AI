import queue
import threading
import time

from loguru import logger
import pyautogui


class ActorThread(threading.Thread):
    def __init__(self, env, action_queue):
        super().__init__(daemon=True)
        self.env = env
        self.action_queue = action_queue
        self.running = True
        self.currently_pressed = set()

    def run(self) -> None:
        while self.running:
            try:
                # get the latest desired action from the Thinker
                action_cmd = self.action_queue.get_nowait()
                desired_keys = self.env.actions[action_cmd.action_type]

                keys_to_release = self.currently_pressed - desired_keys
                for key in keys_to_release:
                    pyautogui.keyUp(key)

                keys_to_press = desired_keys - self.currently_pressed
                for key in keys_to_press:
                    pyautogui.keyDown(key)

                self.currently_pressed = desired_keys

            except queue.Empty:
                time.sleep(0.001)
            except Exception as e:
                logger.error(f"Actor error: {e}")

    def stop(self) -> None:
        for key in self.currently_pressed:
            pyautogui.keyUp(key)
        self.running = False
