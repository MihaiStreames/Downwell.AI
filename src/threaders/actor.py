import queue
import threading
import time

from loguru import logger
import pyautogui


class ActorThread(threading.Thread):
    """Executes actions without blocking perception.

    This thread consumes action commands from the queue and translates
    them into keyboard inputs, managing key press/release states efficiently.

    Parameters
    ----------
    env : CustomDownwellEnvironment
        Environment containing action-to-key mappings.
    action_queue : queue.Queue
        Queue of action commands from thinker thread.

    Attributes
    ----------
    env : CustomDownwellEnvironment
        Game environment.
    action_queue : queue.Queue
        Action command queue.
    running : bool
        Flag to control thread execution.
    currently_pressed : set[str]
        Set of keys currently held down.
    """

    def __init__(self, env, action_queue):
        super().__init__(daemon=True)
        self.env = env
        self.action_queue = action_queue
        self.running = True
        self.currently_pressed = set()

    def run(self) -> None:
        """Main thread loop for executing actions.

        Continuously polls the action queue and translates actions into
        keyboard inputs. Efficiently manages key states by only pressing/
        releasing keys that changed from the previous action.
        """
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

    def stop(self) -> None:
        """Stop the actor thread and release all held keys."""
        # When stopping, release all keys that are currently pressed
        for key in self.currently_pressed:
            pyautogui.keyUp(key)
        self.running = False
