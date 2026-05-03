import threading
import time

from loguru import logger
import pyautogui

from src.utils.consts import ACTION_KEYS
from src.utils.consts import ACTION_NONE


class ActorThread(threading.Thread):
    """Applies key presses based on desired action set by thinker."""

    def __init__(self) -> None:
        super().__init__(daemon=True)

        self._desired_action: int = ACTION_NONE
        self._lock: threading.Lock = threading.Lock()

        self._currently_pressed: set[str] = set()

        self._running: bool = True

    def set_action(self, action: int) -> None:
        with self._lock:
            self._desired_action = action

    def _release_all_keys(self) -> None:
        for key in self._currently_pressed:
            pyautogui.keyUp(key)
        self._currently_pressed = set()

    def run(self) -> None:
        while self._running:
            try:
                with self._lock:
                    action = self._desired_action

                desired_keys = ACTION_KEYS[action]

                keys_to_release = self._currently_pressed - desired_keys
                for key in keys_to_release:
                    pyautogui.keyUp(key)

                keys_to_press = desired_keys - self._currently_pressed
                for key in keys_to_press:
                    pyautogui.keyDown(key)

                if keys_to_release or keys_to_press:
                    self._currently_pressed = set(desired_keys)

            except Exception as e:
                logger.error(f"Actor error: {e}")

            time.sleep(0.005)

    def stop(self) -> None:
        self._release_all_keys()
        self._running = False
