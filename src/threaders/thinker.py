from collections import deque
import threading
import time
from typing import TYPE_CHECKING

from loguru import logger

from src.models.game_state import GameState
from src.utils.consts import ACTION_JUMP
from src.utils.consts import ACTION_LEFT
from src.utils.consts import ACTION_LEFT_JUMP
from src.utils.consts import ACTION_NONE
from src.utils.consts import ACTION_RIGHT
from src.utils.consts import ACTION_RIGHT_JUMP
from src.utils.consts import HP_TRANSITION_SENTINEL
from src.utils.consts import WALL_LEFT
from src.utils.consts import WALL_RIGHT


if TYPE_CHECKING:
    from src.agents.dqn_agent import DQNAgent
    from src.core.reward_calculator import RewardCalculator
    from src.threaders.actor import ActorThread


def _filter_boundary_action(action: int, xpos: float | None) -> int:
    if xpos is None:
        return action

    if xpos <= WALL_LEFT and action == ACTION_LEFT:
        return ACTION_NONE
    if xpos <= WALL_LEFT and action == ACTION_LEFT_JUMP:
        return ACTION_JUMP
    if xpos >= WALL_RIGHT and action == ACTION_RIGHT:
        return ACTION_NONE
    if xpos >= WALL_RIGHT and action == ACTION_RIGHT_JUMP:
        return ACTION_JUMP

    return action


class ThinkerThread(threading.Thread):
    def _reset_episode(self) -> None:
        self._episode_reward = 0.0
        self._experiences_added = 0
        self._step_count = 0
        self._min_ypos = float("inf")
        self._last_state = None
        self._last_action = None
        self._current_reward = 0.0

    def __init__(
        self,
        agent: "DQNAgent",
        reward_calc: "RewardCalculator",
        actor: "ActorThread",
        state_buffer: deque[GameState],
        perceptor_lock: threading.Lock,
        decision_fps: int = 60,
    ) -> None:
        super().__init__(daemon=True)

        self._agent: DQNAgent = agent
        self._reward_calc: RewardCalculator = reward_calc
        self._actor: ActorThread = actor
        self._state_buffer: deque[GameState] = state_buffer
        self._perceptor_lock: threading.Lock = perceptor_lock

        self._decision_interval: float = 1.0 / decision_fps

        self._last_state: GameState | None
        self._last_action: int | None
        self._step_count: int
        self._episode_reward: float
        self._experiences_added: int
        self._min_ypos: float
        self._current_reward: float
        self._reset_episode()

        self._running: bool = True

    @property
    def current_reward(self) -> float:
        return self._current_reward

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

                    is_transition_state = current_state.hp == HP_TRANSITION_SENTINEL

                    if self._last_state is not None and self._last_action is not None:
                        reward = self._reward_calc.calculate_reward(self._last_state, current_state)
                        self._current_reward = reward
                        self._episode_reward += reward

                        if (
                            not is_transition_state
                            and self._last_state.hp != HP_TRANSITION_SENTINEL
                        ):
                            done = current_state.hp is not None and current_state.hp <= 0

                            loss = self._agent.train(
                                self._last_state,
                                self._last_action,
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
                    self._actor.set_action(action)

                    self._last_state = current_state
                    self._last_action = action

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

        self._reset_episode()
        return stats

    def stop(self) -> None:
        self._running = False
