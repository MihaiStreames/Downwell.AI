import queue
import threading
import time
from typing import TYPE_CHECKING

from loguru import logger
import pyautogui

from src.threaders.thinker import Action


if TYPE_CHECKING:
    from src.environment.game_env import CustomDownwellEnvironment


class ActorThread(threading.Thread):
    def __init__(self, env: "CustomDownwellEnvironment", action_queue: queue.Queue[Action]) -> None:
        super().__init__(daemon=True)

        self._env: CustomDownwellEnvironment = env
        self._action_queue: queue.Queue[Action] = action_queue

        self._running: bool = True
        self._currently_pressed: set[str] = set()

    def run(self) -> None:
        while self._running:
            try:
                action_cmd = self._action_queue.get_nowait()
                desired_keys = self._env.actions[action_cmd.action_type]

                keys_to_release = self._currently_pressed - desired_keys
                for key in keys_to_release:
                    pyautogui.keyUp(key)

                keys_to_press = desired_keys - self._currently_pressed
                for key in keys_to_press:
                    pyautogui.keyDown(key)

                self._currently_pressed = desired_keys

            except queue.Empty:
                time.sleep(0.001)
            except Exception as e:
                logger.error(f"Actor error: {e}")

    def stop(self) -> None:
        for key in self._currently_pressed:
            pyautogui.keyUp(key)
        self._running = False
