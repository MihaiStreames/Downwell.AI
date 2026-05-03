from collections import deque
import queue
import threading
import time
from typing import TYPE_CHECKING
from typing import NamedTuple

from loguru import logger

from src.models.game_state import GameState


if TYPE_CHECKING:
    from src.agents.dqn_agent import DQNAgent
    from src.core.reward_calculator import RewardCalculator


class Action(NamedTuple):
    action_type: int
    frame_id: int


def _filter_boundary_action(action: int, xpos: float | None) -> int:
    if xpos is None:
        return action

    left_boundary = 180
    right_boundary = 300

    if xpos <= left_boundary and action == 2:  # left -> no-op
        return 0
    if xpos <= left_boundary and action == 4:  # left+jump -> jump
        return 1
    if xpos >= right_boundary and action == 3:  # right -> no-op
        return 0
    if xpos >= right_boundary and action == 5:  # right+jump -> jump
        return 1

    return action


class ThinkerThread(threading.Thread):
    def __init__(
        self,
        agent: "DQNAgent",
        reward_calc: "RewardCalculator",
        state_buffer: deque[GameState],
        action_queue: queue.Queue[Action],
        perceptor_lock: threading.Lock,
        decision_fps: int = 60,
    ) -> None:
        super().__init__(daemon=True)

        self._agent: DQNAgent = agent
        self._action_queue: queue.Queue[Action] = action_queue
        self._perceptor_lock: threading.Lock = perceptor_lock
        self._decision_interval: float = 1.0 / decision_fps

        self._reward_calc: RewardCalculator = reward_calc
        self._state_buffer: deque[GameState] = state_buffer

        self._last_state: GameState | None = None
        self._last_action: Action | None = None

        self._step_count: int = 0
        self.current_reward: float = 0.0
        self._episode_reward: float = 0.0
        self._experiences_added: int = 0
        self._min_ypos: float = float("inf")

        self._running: bool = True

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

                    action, q_values = self._agent.get_action(current_state)
                    action = _filter_boundary_action(action, current_state.xpos)
                    action_cmd = Action(action_type=action, frame_id=current_state.frame_id)

                    # drain stale actions before pushing new one to prevent queue lag
                    # from causing boundary-filtered actions to arrive too late
                    while not self._action_queue.empty():
                        try:
                            self._action_queue.get_nowait()
                        except queue.Empty:
                            break

                    self._action_queue.put(action_cmd)

                    self._last_state = current_state
                    self._last_action = action_cmd

            except Exception as e:
                logger.error(f"Thinker error: {e}")

            elapsed = time.time() - start_time
            sleep_time = self._decision_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_episode_stats(self) -> dict[str, float | int]:
        stats: dict[str, float | int] = {
            "episode_reward": self._episode_reward,
            "experiences_added": self._experiences_added,
            "steps": self._step_count,
            "max_ypos_reached": self._min_ypos if self._min_ypos != float("inf") else 0.0,
            "level_reached": self._reward_calc.current_level,
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
