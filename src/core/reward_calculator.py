from src.config import Config
from src.models.game_state import GameState


class RewardCalculator:
    def __init__(self, config: Config):
        self.config = config

        self.max_depth_achieved: float = 0.0

        self.last_hp: float | None = None
        self.last_gems: float = 0.0
        self.last_combo: float = 0.0

    @staticmethod
    def _detect_level_completion(state: GameState, next_state: GameState) -> bool:
        if state is None or state.xpos is None or state.ypos is None:
            return False

        player_in_well = state.ypos < -100
        player_in_menu = next_state.xpos is None or next_state.hp is None
        return player_in_well and player_in_menu

    @staticmethod
    def calculate_boundary_penalty(xpos: float | None) -> float:
        if xpos is None:
            return 0.0

        # safe zone
        if 172 <= xpos <= 308:
            return 0.0

        # out of bounds
        if xpos < 172:
            # left out of bounds
            distance_out = 172 - xpos
            return -1.0 * (1 + distance_out * 0.1)

        if xpos > 308:
            # right out of bounds
            distance_out = xpos - 308
            return -1.0 * (1 + distance_out * 0.1)

        return 0.0

    def calculate_reward(self, state: GameState, next_state: GameState) -> float:
        # level completion bonus
        if self._detect_level_completion(state, next_state):
            self.reset_episode()
            return self.config.level_complete_bonus

        if state.hp == 999.0 or next_state.hp == 999.0:
            return 0.0

        # death penalty
        if state.hp is not None and state.hp > 0 and (next_state.hp is None or next_state.hp <= 0):
            return float(self.config.death_penalty)

        # no reward if already dead
        if next_state.hp is None or next_state.hp <= 0:
            return 0.0

        # default step penalty
        reward = self.config.step_penalty

        # main reward: going deeper
        if (
            state.ypos is not None
            and next_state.ypos is not None
            and next_state.ypos < self.max_depth_achieved
        ):
            progress = self.max_depth_achieved - next_state.ypos
            reward += progress * self.config.depth_reward
            self.max_depth_achieved = next_state.ypos

        # gem reward
        gems_collected = next_state.gems - self.last_gems
        if gems_collected > 0:
            reward += gems_collected * self.config.gem_reward
        self.last_gems = next_state.gems

        # combo bonus
        if next_state.combo > self.config.combo_threshold:
            reward += self.config.combo_bonus_multiplier * next_state.combo
        self.last_combo = next_state.combo

        # damage penalty
        if self.last_hp is not None and next_state.hp is not None:
            damage_taken = self.last_hp - next_state.hp
            if damage_taken > 0:
                damage_penalty = damage_taken * self.config.damage_penalty
                reward += damage_penalty

        # update HP tracking
        if next_state.hp is not None:
            self.last_hp = next_state.hp

        # boundary penalty
        boundary_penalty = self.calculate_boundary_penalty(next_state.xpos)
        reward += boundary_penalty

        # clip the reward
        return float(max(self.config.min_reward_clip, min(reward, self.config.max_reward_clip)))

    def reset_episode(self) -> None:
        self.max_depth_achieved = 0.0
        self.last_hp = None
        self.last_gems = 0.0
        self.last_combo = 0.0
