import queue
import time
from collections import deque
from typing import Optional

from models.game_state import GameState
from threaders.actor import ActorThread
from threaders.perceptor import PerceptorThread
from threaders.thinker import ThinkerThread


class DownwellAI:
    """Main orchestrator for the AI system"""

    def __init__(self, player, env, agent, reward_calculator):
        self.player = player
        self.env = env
        self.agent = agent
        self.reward_calc = reward_calculator

        # Recreated each episode
        self.perceptor = None
        self.thinker = None
        self.actor = None
        self.state_buffer = None
        self.action_queue = None

    def create_threads(self):
        self.state_buffer = deque(maxlen=120)
        self.action_queue = queue.Queue()

        self.perceptor = PerceptorThread(self.player, self.env, self.state_buffer)
        self.thinker = ThinkerThread(self.agent, self.state_buffer, self.action_queue, self.perceptor.lock)
        self.actor = ActorThread(self.env, self.action_queue)

    def start(self):
        self.create_threads()
        self.perceptor.start()
        self.thinker.start()
        self.actor.start()

    def stop(self):
        if self.perceptor: self.perceptor.stop()
        if self.thinker: self.thinker.stop()
        if self.actor: self.actor.stop()

        # Wait for threads to finish
        time.sleep(0.2)

    def get_latest_state(self) -> Optional[GameState]:
        if self.perceptor and self.state_buffer:
            with self.perceptor.lock:
                return self.state_buffer[-1] if self.state_buffer else None
        return None

    def train_on_episode(self):
        if not self.thinker:
            return 0, 0

        episode_data = self.thinker.get_episode_data()
        if not episode_data:
            return 0, 0

        total_reward = 0
        experiences_added = 0

        for experience in episode_data:
            state = experience['state']
            action = experience['action']
            next_state = experience['next_state']

            # Extract memory features for both states
            memory_features = self.agent.extract_memory_features(state)
            next_memory_features = self.agent.extract_memory_features(next_state)

            # Calculate reward based on state transition
            reward = self.reward_calc.calculate_reward(state, next_state)

            # Dead if HP <= 0 and not sentinel value
            done = next_state.hp <= 0 and next_state.hp != 999.0

            self.agent.remember(
                state.screenshot, action.action_type, action.duration,
                reward, next_state.screenshot, done,
                memory_features, next_memory_features
            )

            experiences_added += 1
            total_reward += reward

        # Train if enough experiences
        if len(self.agent.memory) >= self.agent.train_start:
            self.agent.replay()

        return total_reward, experiences_added
