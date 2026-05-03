from src.config import Config
from src.models.game_state import GameState


def _detect_level_completion(state: GameState, next_state: GameState) -> bool:
    if state is None or state.xpos is None or state.ypos is None:
        return False

    player_in_well = state.ypos < -100
    player_in_menu = next_state.xpos is None or next_state.hp is None

    return player_in_well and player_in_menu


def _calculate_boundary_penalty(xpos: float | None) -> float:
    if xpos is None:
        return 0.0

    left_danger = 188
    right_danger = 292
    if left_danger <= xpos <= right_danger:
        return 0.0

    # linear ramp
    if xpos < left_danger:
        distance_in = left_danger - xpos  # 0..28+ as agent nears wall at 160
        return -0.1 * (1 + distance_in * 0.2)
    if xpos > right_danger:
        distance_in = xpos - right_danger
        return -0.1 * (1 + distance_in * 0.2)

    return 0.0


def _calculate_center_reward(xpos: float | None) -> float:
    if xpos is None:
        return 0.0

    center = 240.0
    distance_from_center = abs(xpos - center)

    return -distance_from_center * 0.001


class RewardCalculator:
    def __init__(self, config: Config):
        self._config = config

        self._current_level: int = 1
        self._level_entry_ypos: float | None = None
        self._max_depth_delta: float = 0.0

        self._last_hp: float | None = None
        self._last_gems: float = 0.0
        self._last_combo: float = 0.0
        self._last_ammo: float = 0.0

    @property
    def current_level(self) -> int:
        return self._current_level

    def _reset_episode(self) -> None:
        self._level_entry_ypos = None
        self._max_depth_delta = 0.0
        self._last_hp = None
        self._last_gems = 0.0
        self._last_combo = 0.0
        self._last_ammo = 0.0

    def calculate_reward(self, state: GameState, next_state: GameState) -> float:
        if _detect_level_completion(state, next_state):
            self._current_level += 1
            self._reset_episode()
            return self._config.level_complete_bonus

        # ignore transitions
        if state.hp == 999.0 or next_state.hp == 999.0:
            return 0.0

        # death penalty
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

        combo_delta = next_state.combo - self._last_combo
        if combo_delta > 0:
            reward += combo_delta * self._config.combo_delta_multiplier
        self._last_combo = next_state.combo

        if self._last_hp is not None and next_state.hp is not None:
            damage_taken = self._last_hp - next_state.hp
            if damage_taken > 0:
                reward += damage_taken * self._config.damage_penalty

        if next_state.hp is not None:
            self._last_hp = next_state.hp

        # avoid edges; action filter in thinker still prevents wall crossing
        reward += _calculate_boundary_penalty(next_state.xpos)

        # continuous gradient toward center
        reward += _calculate_center_reward(next_state.xpos)

        return float(max(self._config.min_reward_clip, min(reward, self._config.max_reward_clip)))

    def reset_run(self) -> None:
        self._current_level = 1
        self._reset_episode()
