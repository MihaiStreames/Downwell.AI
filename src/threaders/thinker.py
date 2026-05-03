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

        self._agent = agent
        self._action_queue = action_queue
        self._perceptor_lock = perceptor_lock
        self._decision_interval = 1.0 / decision_fps

        self._reward_calc = reward_calc
        self._state_buffer = state_buffer

        self._last_state = None
        self._last_action = None

        self._step_count = 0
        self.current_reward = 0.0
        self._episode_reward = 0.0
        self._experiences_added = 0
        self._min_ypos = float("inf")

        self._running = True

    def run(self) -> None:
        while self._running:
            start_time = time.time()

            try:
                with self._perceptor_lock:
                    current_state = self._state_buffer[-1] if self._state_buffer else None

                if current_state and current_state.screenshot is not None:
                    self._step_count += 1

                    if current_state.ypos is not None:
                        self._min_ypos = min(self._min_ypos, current_state.ypos)

                    is_transition_state = current_state.hp == 999.0

                    if self._last_state is not None and self._last_action is not None:
                        reward = self._reward_calc.calculate_reward(self._last_state, current_state)
                        self.current_reward = reward
                        self._episode_reward += reward

                        if not is_transition_state and self._last_state.hp != 999.0:
                            done = current_state.hp is not None and current_state.hp <= 0

                            loss = self._agent.train(
                                self._last_state,
                                self._last_action.action_type,
                                reward,
                                current_state,
                                done,
                            )
                            self._experiences_added += 1

                            if loss is not None and self._step_count % 100 == 0:
                                logger.debug(
                                    f"Step {self._step_count}: Loss = {loss:.4f}, Reward = {reward:.2f}"
                                )

                    # make decision
                    action, q_values = self._agent.get_action(current_state)
                    action_cmd = Action(action_type=action, frame_id=current_state.frame_id)
                    self._action_queue.put(action_cmd)

                    self._last_state = current_state
                    self._last_action = action_cmd

            except Exception as e:
                logger.error(f"Thinker error: {e}")

            # maintain decision rate
            elapsed = time.time() - start_time
            sleep_time = self._decision_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_episode_stats(self) -> dict:
        stats = {
            "episode_reward": self._episode_reward,
            "experiences_added": self._experiences_added,
            "steps": self._step_count,
            "max_ypos_reached": self._min_ypos if self._min_ypos != float("inf") else 0.0,
        }

        self._episode_reward = 0.0
        self._experiences_added = 0
        self._step_count = 0
        self._min_ypos = float("inf")
        self._last_state = None
        self._last_action = None

        return stats

    def stop(self) -> None:
        self._running = False
