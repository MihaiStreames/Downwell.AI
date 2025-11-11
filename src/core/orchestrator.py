import queue
import time
from collections import deque
from typing import Optional

from config import EnvConfig
from models.game_state import GameState
from threaders.actor import ActorThread
from threaders.perceptor import PerceptorThread
from threaders.thinker import ThinkerThread


class DownwellAI:
    """Main orchestrator for the AI system"""

    def __init__(self, player, env, agent, reward_calculator, config: EnvConfig):
        self.player = player
        self.env = env
        self.agent = agent
        self.reward_calc = reward_calculator
        self.config = config

        # Recreated each episode
        self.perceptor = None
        self.thinker = None
        self.actor = None
        self.state_buffer = None
        self.action_queue = None

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

    def get_latest_state(self) -> Optional[GameState]:
        if self.perceptor and self.state_buffer:
            with self.perceptor.lock:
                return self.state_buffer[-1] if self.state_buffer else None
        return None

    def get_episode_stats(self):
        if self.thinker:
            return self.thinker.get_episode_stats()
        return {"episode_reward": 0.0, "experiences_added": 0, "steps": 0}
