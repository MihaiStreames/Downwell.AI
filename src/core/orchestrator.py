from collections import deque
import queue
import time
from typing import TYPE_CHECKING

from src.config import Config
from src.core.reward_calculator import RewardCalculator
from src.models.game_state import GameState
from src.threaders.actor import ActorThread
from src.threaders.perceptor import PerceptorThread
from src.threaders.thinker import ThinkerThread


if TYPE_CHECKING:
    from src.agents.dqn_agent import DQNAgent
    from src.environment.game_env import CustomDownwellEnvironment
    from src.environment.mem_extractor import Player


class DownwellAI:
    def __init__(
        self,
        player: "Player",
        env: "CustomDownwellEnvironment",
        agent: "DQNAgent",
        reward_calculator: "RewardCalculator",
        config: "Config",
    ) -> None:
        self._player: Player = player
        self._env: CustomDownwellEnvironment = env
        self._agent: DQNAgent = agent
        self._reward_calc: RewardCalculator = reward_calculator
        self._config: Config = config

        self._perceptor: PerceptorThread | None = None
        self._thinker: ThinkerThread | None = None
        self._actor: ActorThread | None = None
        self._state_buffer: deque[GameState] | None = None
        self._action_queue: queue.Queue | None = None

    @property
    def thinker(self) -> ThinkerThread | None:
        return self._thinker

    def start(self) -> None:
        self._state_buffer = deque(maxlen=120)
        self._action_queue = queue.Queue()

        self._perceptor = PerceptorThread(
            self._player,
            self._env,
            self._state_buffer,
            perception_fps=self._config.perceptor_fps,
        )
        self._thinker = ThinkerThread(
            self._agent,
            self._reward_calc,
            self._state_buffer,
            self._action_queue,
            self._perceptor.lock,
            decision_fps=self._config.thinker_fps,
        )
        self._actor = ActorThread(self._env, self._action_queue)

        self._perceptor.start()
        self._thinker.start()
        self._actor.start()
        self._reward_calc.reset_run()

    def stop(self) -> None:
        if self._perceptor:
            self._perceptor.stop()

        if self._thinker:
            self._thinker.stop()

        if self._actor:
            self._actor.stop()

        time.sleep(0.2)

    def get_latest_state(self) -> GameState | None:
        if self._perceptor and self._state_buffer:
            with self._perceptor.lock:
                return self._state_buffer[-1] if self._state_buffer else None
        return None

    def get_episode_stats(self) -> dict[str, float | int]:
        if self._thinker:
            return self._thinker.get_episode_stats()
        return {"episode_reward": 0.0, "experiences_added": 0, "steps": 0}
