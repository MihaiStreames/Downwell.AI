import queue
import threading
import time

import vgamepad as vg
from loguru import logger


class VGamepadActor(threading.Thread):
    """Executes actions using virtual Xbox 360 controller without requiring window focus"""

    def __init__(self, env, action_queue):
        super().__init__(daemon=True)
        self.env = env
        self.action_queue = action_queue
        self.running = True

        # Initialize virtual Xbox 360 controller
        try:
            self.gamepad = vg.VX360Gamepad()
            logger.debug("Virtual Xbox 360 controller initialized")
        except Exception as e:
            logger.error(f"Failed to initialize vgamepad: {e}")
            logger.error("Make sure ViGEmBus driver is installed")
            raise

        # Map action indices to controller functions
        self.action_map = self._create_action_map()

        # Track current state to avoid redundant updates
        self.current_action = 0

        # Reset controller to neutral state
        self._reset_controller()

    def _create_action_map(self) -> dict[int, callable]:
        return {
            0: self._action_none,
            1: self._action_jump,
            2: self._action_left,
            3: self._action_right,
            4: self._action_left_jump,
            5: self._action_right_jump,
        }

    def _reset_controller(self):
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)

        self.gamepad.left_joystick(x_value=0, y_value=0)
        self.gamepad.right_joystick(x_value=0, y_value=0)

        self.gamepad.update()

        logger.debug("Controller reset to neutral state")

    def _action_none(self):
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
        self.gamepad.update()

    def _action_jump(self):
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
        self.gamepad.update()

    def _action_left(self):
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
        self.gamepad.update()

    def _action_right(self):
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
        self.gamepad.update()

    def _action_left_jump(self):
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
        self.gamepad.update()

    def _action_right_jump(self):
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
        self.gamepad.update()

    def run(self):
        while self.running:
            try:
                # Get the latest desired action from the Thinker
                action_cmd = self.action_queue.get_nowait()

                # Only update if action changed (reduces USB traffic)
                if action_cmd.action_type != self.current_action:
                    self.action_map[action_cmd.action_type]()
                    self.current_action = action_cmd.action_type

            except queue.Empty:
                time.sleep(0.001)  # Sleep briefly if no new action
            except Exception as e:
                logger.error(f"VGamepad Actor error: {e}")

    def stop(self):
        self.running = False
        self._reset_controller()

        try:
            self.gamepad.reset()
            self.gamepad.update()
        except Exception as e:
            logger.error(f"Failed to reset vgamepad on stop: {e}")
            pass
