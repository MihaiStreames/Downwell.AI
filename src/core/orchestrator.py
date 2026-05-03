from collections import deque
import queue
import time

from src.config import Config
from src.models.game_state import GameState
from src.threaders.actor import ActorThread
from src.threaders.perceptor import PerceptorThread
from src.threaders.thinker import ThinkerThread


class DownwellAI:
    def __init__(self, player, env, agent, reward_calculator, config: Config):
        self._player = player
        self._env = env
        self._agent = agent
        self._reward_calc = reward_calculator
        self._config = config

        # recreated each episode
        self._perceptor: PerceptorThread | None = None
        self.thinker: ThinkerThread | None = None
        self._actor: ActorThread | None = None
        self._state_buffer: deque | None = None
        self._action_queue: queue.Queue | None = None

    def start(self) -> None:
        self._state_buffer = deque(maxlen=120)
        self._action_queue = queue.Queue()

        self._perceptor = PerceptorThread(
            self._player,
            self._env,
            self._state_buffer,
            perception_fps=self._config.perceptor_fps,
        )
        self.thinker = ThinkerThread(
            self._agent,
            self._reward_calc,
            self._state_buffer,
            self._action_queue,
            self._perceptor.lock,
            decision_fps=self._config.thinker_fps,
        )
        self._actor = ActorThread(self._env, self._action_queue)

        self._perceptor.start()
        self.thinker.start()
        self._actor.start()
        self._reward_calc.reset_episode()

    def stop(self) -> None:
        if self._perceptor:
            self._perceptor.stop()

        if self.thinker:
            self.thinker.stop()

        if self._actor:
            self._actor.stop()

        time.sleep(0.2)

    def get_latest_state(self) -> GameState | None:
        if self._perceptor and self._state_buffer:
            with self._perceptor.lock:
                return self._state_buffer[-1] if self._state_buffer else None
        return None

    def get_episode_stats(self) -> dict:
        if self.thinker:
            return self.thinker.get_episode_stats()
        return {"episode_reward": 0.0, "experiences_added": 0, "steps": 0}
