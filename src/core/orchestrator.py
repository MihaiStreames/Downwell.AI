from collections import deque
import queue
import time

from src.config import EnvConfig
from src.models.game_state import GameState
from src.threaders.actor import ActorThread
from src.threaders.perceptor import PerceptorThread
from src.threaders.thinker import ThinkerThread


class DownwellAI:
    """Main orchestrator for the AI system"""

    def __init__(self, player, env, agent, reward_calculator, config: EnvConfig):
        self.player = player
        self.env = env
        self.agent = agent
        self.reward_calc = reward_calculator
        self.config = config

        # Recreated each episode
        self.perceptor: PerceptorThread | None = None
        self.thinker: ThinkerThread | None = None
        self.actor: ActorThread | None = None
        self.state_buffer: deque | None = None
        self.action_queue: queue.Queue | None = None

    def create_threads(self):
        self.state_buffer = deque(maxlen=120)
        self.action_queue = queue.Queue()

        self.perceptor = PerceptorThread(
            self.player,
            self.env,
            self.state_buffer,
            perception_fps=self.config.perceptor_fps,
        )
        self.thinker = ThinkerThread(
            self.agent,
            self.reward_calc,
            self.state_buffer,
            self.action_queue,
            self.perceptor.lock,
            decision_fps=self.config.thinker_fps,
        )
        self.actor = ActorThread(self.env, self.action_queue)

    def start(self):
        self.create_threads()
        self.perceptor.start()
        self.thinker.start()
        self.actor.start()

        # Reset reward calculator for new episode
        self.reward_calc.reset_episode()

    def stop(self):
        if self.perceptor:
            self.perceptor.stop()
        if self.thinker:
            self.thinker.stop()
        if self.actor:
            self.actor.stop()
        # Wait for threads to finish
        time.sleep(0.2)

    def get_latest_state(self) -> GameState | None:
        if self.perceptor and self.state_buffer:
            with self.perceptor.lock:
                return self.state_buffer[-1] if self.state_buffer else None
        return None

    def get_episode_stats(self):
        if self.thinker:
            return self.thinker.get_episode_stats()
        return {"episode_reward": 0.0, "experiences_added": 0, "steps": 0}
