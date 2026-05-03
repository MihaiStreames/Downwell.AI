import threading
import time

from loguru import logger

from src.models.game_state import GameState


class PerceptorThread(threading.Thread):
    def __init__(self, player, env, state_buffer, perception_fps=60):
        super().__init__(daemon=True)
        self.lock = threading.Lock()

        self._player = player
        self._env = env

        self._state_buffer = state_buffer

        self._target_fps = perception_fps
        self._frame_interval = 1.0 / perception_fps
        self._frame_count = 0

        self._running = True

    def run(self) -> None:
        while self._running:
            start_time = time.time()

            try:
                screenshot = self._env.get_state()

                hp = self._player.get_value("hp")
                xpos = self._player.get_value("xpos")
                ypos = self._player.get_value("ypos")

                is_transition_state = xpos is None or hp is None

                # if in a transition, set a sentinel value for HP for other parts of the system,
                # but keep xpos/ypos as None to be used for level detection
                if is_transition_state:
                    hp = 999.0

                gems = self._player.get_value("gems") or 0
                combo = self._player.get_value("combo") or 0
                ammo = self._player.get_value("ammo") or 0
                gem_high = self._player.is_gem_high()

                state = GameState(
                    screenshot=screenshot,
                    hp=hp,
                    gems=gems,
                    combo=combo,
                    xpos=xpos,
                    ypos=ypos,
                    ammo=ammo,
                    gem_high=gem_high,
                    timestamp=time.time(),
                    frame_id=self._frame_count,
                )

                with self.lock:
                    self._state_buffer.append(state)

                    self._frame_count += 1

            except Exception as e:
                logger.error(f"Perceptor error: {e}")

            elapsed = time.time() - start_time
            sleep_time = self._frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self) -> None:
        self._running = False
