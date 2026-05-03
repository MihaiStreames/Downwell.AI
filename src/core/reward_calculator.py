from src.config import Config
from src.models.game_state import GameState
from src.utils.consts import CENTER_XPOS
from src.utils.consts import DANGER_LEFT
from src.utils.consts import DANGER_RIGHT
from src.utils.consts import HP_TRANSITION_SENTINEL
from src.utils.consts import LEVEL_COMPLETE_YPOS


def _detect_level_completion(state: GameState, next_state: GameState) -> bool:
    if state is None or state.xpos is None or state.ypos is None:
        return False

    player_in_well = state.ypos < LEVEL_COMPLETE_YPOS
    player_in_menu = next_state.xpos is None or next_state.hp is None

    return player_in_well and player_in_menu


def _calculate_boundary_penalty(xpos: float | None, base: float, ramp: float) -> float:
    if xpos is None:
        return 0.0

    if DANGER_LEFT <= xpos <= DANGER_RIGHT:
        return 0.0

    # linear ramp
    if xpos < DANGER_LEFT:
        distance_in = DANGER_LEFT - xpos
        return base * (1 + distance_in * ramp)
    if xpos > DANGER_RIGHT:
        distance_in = xpos - DANGER_RIGHT
        return base * (1 + distance_in * ramp)

    return 0.0


def _calculate_center_reward(xpos: float | None, coefficient: float) -> float:
    if xpos is None:
        return 0.0

    distance_from_center = abs(xpos - CENTER_XPOS)
    return -distance_from_center * coefficient


class RewardCalculator:
    def _reset_episode(self) -> None:
        self._level_entry_ypos = None
        self._max_depth_delta = 0.0
        self._last_hp = None
        self._last_gems = 0.0
        self._last_ammo = 0.0

    def __init__(self, config: Config):
        self._config: Config = config
        self._current_level: int = 1

        self._level_entry_ypos: float | None
        self._max_depth_delta: float
        self._last_hp: float | None
        self._last_gems: float
        self._last_ammo: float
        self._reset_episode()

    @property
    def current_level(self) -> int:
        return self._current_level

    def calculate_reward(self, state: GameState, next_state: GameState) -> float:
        if _detect_level_completion(state, next_state):
            self._current_level += 1
            self._reset_episode()
            return self._config.level_complete_bonus

        if HP_TRANSITION_SENTINEL in (state.hp, next_state.hp):
            return 0.0

        if state.hp is not None and state.hp > 0 and (next_state.hp is None or next_state.hp <= 0):
            return float(self._config.death_penalty)
        if next_state.hp is None or next_state.hp <= 0:
            return 0.0

        # establish level entry reference on first valid ypos after reset/transition
        if self._level_entry_ypos is None and next_state.ypos is not None:
            self._level_entry_ypos = next_state.ypos

        reward = self._config.step_penalty

        # per-level delta so rooms don't earn ongoing reward
        if (
            self._level_entry_ypos is not None
            and state.ypos is not None
            and next_state.ypos is not None
        ):
            depth_delta = next_state.ypos - self._level_entry_ypos  # negative = deeper
            if depth_delta < self._max_depth_delta:
                progress = self._max_depth_delta - depth_delta
                reward += progress * self._config.depth_reward
                self._max_depth_delta = depth_delta

        gems_collected = next_state.gems - self._last_gems
        if gems_collected > 0:
            reward += gems_collected * self._config.gem_reward
        self._last_gems = next_state.gems

        ammo_delta = self._last_ammo - next_state.ammo
        if ammo_delta > 0 and next_state.combo > 0:
            reward += ammo_delta * self._config.ammo_kill_reward
        self._last_ammo = next_state.ammo

        if next_state.combo > self._config.combo_threshold:
            reward += self._config.combo_bonus_multiplier * next_state.combo

        if self._last_hp is not None and next_state.hp is not None:
            damage_taken = self._last_hp - next_state.hp
            if damage_taken > 0:
                reward += damage_taken * self._config.damage_penalty

        if next_state.hp is not None:
            self._last_hp = next_state.hp

        # avoid edges
        reward += _calculate_boundary_penalty(
            next_state.xpos,
            self._config.boundary_penalty_base,
            self._config.boundary_penalty_ramp,
        )

        # continuous gradient toward center
        reward += _calculate_center_reward(next_state.xpos, self._config.center_pull_coefficient)

        return float(max(self._config.min_reward_clip, min(reward, self._config.max_reward_clip)))

    def reset_run(self) -> None:
        self._current_level = 1
        self._reset_episode()
