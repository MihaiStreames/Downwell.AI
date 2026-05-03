from collections import deque
import threading
import time
from typing import TYPE_CHECKING

from loguru import logger

from src.models.game_state import GameState


if TYPE_CHECKING:
    from src.environment.game_env import CustomDownwellEnvironment
    from src.environment.mem_extractor import Player


class PerceptorThread(threading.Thread):
    def __init__(
        self,
        player: "Player",
        env: "CustomDownwellEnvironment",
        state_buffer: deque[GameState],
        perception_fps: int = 60,
    ) -> None:
        super().__init__(daemon=True)
        self._lock: threading.Lock = threading.Lock()

        self._player: Player = player
        self._env: CustomDownwellEnvironment = env
        self._state_buffer: deque[GameState] = state_buffer

        self._target_fps: int = perception_fps
        self._frame_interval: float = 1.0 / perception_fps
        self._frame_count: int = 0
        self._running: bool = True

    @property
    def lock(self) -> threading.Lock:
        return self._lock

    def run(self) -> None:
        while self._running:
            start_time = time.time()

            try:
                screenshot = self._env.get_state()

                hp = self._player.get_value("hp")

                # during level transitions hp becomes unreadable
                # skip xpos/ypos reads to avoid flooding logs with pointer-chain failures at 60fps
                if hp is None:
                    xpos = None
                    ypos = None
                else:
                    xpos = self._player.get_value("xpos")
                    ypos = self._player.get_value("ypos")

                is_transition_state = hp is None or xpos is None

                # sentinel HP so reward_calculator ignores transition frames
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
