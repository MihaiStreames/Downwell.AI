import threading
import time
from typing import NamedTuple

from loguru import logger


class Action(NamedTuple):
    action_type: int
    frame_id: int


class ThinkerThread(threading.Thread):
    def __init__(
        self,
        agent,
        reward_calc,
        state_buffer,
        action_queue,
        perceptor_lock,
        decision_fps=60,
    ):
        super().__init__(daemon=True)
        self.agent = agent
        self.reward_calc = reward_calc
        self.state_buffer = state_buffer
        self.action_queue = action_queue
        self.perceptor_lock = perceptor_lock
        self.decision_interval = 1.0 / decision_fps
        self.running = True

        self.last_state = None
        self.last_action = None
        self.step_count = 0
        self.episode_reward = 0.0
        self.current_reward = 0.0
        self.experiences_added = 0

    def run(self) -> None:
        while self.running:
            start_time = time.time()

            try:
                with self.perceptor_lock:
                    current_state = self.state_buffer[-1] if self.state_buffer else None

                if current_state and current_state.screenshot is not None:
                    self.step_count += 1
                    is_transition_state = current_state.hp == 999.0

                    if self.last_state is not None and self.last_action is not None:
                        reward = self.reward_calc.calculate_reward(self.last_state, current_state)
                        self.current_reward = reward
                        self.episode_reward += reward

                        if not is_transition_state and self.last_state.hp != 999.0:
                            done = current_state.hp is not None and current_state.hp <= 0

                            loss = self.agent.train(
                                self.last_state,
                                self.last_action.action_type,
                                reward,
                                current_state,
                                done,
                            )
                            self.experiences_added += 1

                            if loss is not None and self.step_count % 100 == 0:
                                logger.debug(
                                    f"Step {self.step_count}: Loss = {loss:.4f}, Reward = {reward:.2f}"
                                )

                    # make decision
                    action, q_values = self.agent.get_action(current_state)
                    action_cmd = Action(action_type=action, frame_id=current_state.frame_id)
                    self.action_queue.put(action_cmd)

                    self.last_state = current_state
                    self.last_action = action_cmd

            except Exception as e:
                logger.error(f"Thinker error: {e}")

            # maintain decision rate
            elapsed = time.time() - start_time
            sleep_time = self.decision_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_episode_stats(self) -> dict:
        stats = {
            "episode_reward": self.episode_reward,
            "experiences_added": self.experiences_added,
            "steps": self.step_count,
        }

        self.episode_reward = 0.0
        self.experiences_added = 0
        self.step_count = 0
        self.last_state = None
        self.last_action = None

        return stats

    def stop(self) -> None:
        self.running = False
