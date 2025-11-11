from loguru import logger

from config import RewardConfig
from models.game_state import GameState


class RewardCalculator:
    """Reward system for Downwell.AI"""

    def __init__(self, config: RewardConfig):
        self.config = config
        self.max_depth_achieved = 0.0
        self.steps_without_progress = 0
        self.last_progress_depth = 0.0
        self.last_hp = None

    @staticmethod
    def _detect_level_completion(state: GameState, next_state: GameState) -> bool:
        if state is None or state.xpos is None or state.ypos is None:
            return False
        player_was_in_well = state.ypos < -100
        player_is_in_menu = next_state.xpos is None or next_state.hp is None
        return player_was_in_well and player_is_in_menu

    def calculate_reward(self, state: GameState, next_state: GameState) -> float:
        # Level completion bonus
        if self._detect_level_completion(state, next_state):
            logger.debug(f"LEVEL COMPLETE! Reward: +{self.config.level_complete_bonus}")
            self.reset_episode()
            return self.config.level_complete_bonus

        if state.hp == 999.0 or next_state.hp == 999.0:
            return 0.0

        # Death penalty
        if state.hp > 0 and (next_state.hp is None or next_state.hp <= 0):
            return self.config.death_penalty

        # No reward if already dead
        if next_state.hp is None or next_state.hp <= 0:
            return 0.0

        reward = 0.0

        # Main reward: Going deeper
        if state.ypos is not None and next_state.ypos is not None:
            if next_state.ypos < self.max_depth_achieved:
                progress = self.max_depth_achieved - next_state.ypos
                reward += progress * self.config.depth_reward
                self.max_depth_achieved = next_state.ypos

                # Reset immobility counter on progress
                self.steps_without_progress = 0
                self.last_progress_depth = next_state.ypos
            else:
                # No progress made, increment counter
                self.steps_without_progress += 1

        # Damage penalty
        if self.last_hp is not None and next_state.hp is not None:
            damage_taken = self.last_hp - next_state.hp
            if damage_taken > 0:
                # Apply penalty based on amount of damage
                damage_penalty = damage_taken * self.config.damage_penalty
                reward += damage_penalty  # This is negative
                logger.debug(
                    f"Took {damage_taken:.0f} damage! Penalty: {damage_penalty:.2f}"
                )

        # Update HP tracking
        self.last_hp = next_state.hp

        # Exponential immobility penalty that increases the longer you don't progress
        if self.steps_without_progress > self.config.immobility_grace_period:
            # Exponential penalty after grace period
            steps_stagnant = (
                self.steps_without_progress - self.config.immobility_grace_period
            )
            immobility_penalty = -self.config.immobility_base_penalty * (
                self.config.immobility_growth_rate**steps_stagnant
            )

            # Cap the penalty to prevent it from becoming too extreme
            immobility_penalty = max(
                immobility_penalty, -self.config.immobility_max_penalty
            )
            reward += immobility_penalty

            # Log every 30 steps of immobility
            if self.steps_without_progress % 30 == 0:
                logger.debug(
                    f"No progress for {self.steps_without_progress} steps! Penalty: {immobility_penalty:.2f}"
                )

        # Small survival reward to encourage staying alive
        reward += self.config.survival_reward

        # Clip the reward
        return max(
            self.config.min_reward_clip, min(reward, self.config.max_reward_clip)
        )

    def reset_episode(self):
        self.max_depth_achieved = 0.0
        self.steps_without_progress = 0
        self.last_progress_depth = 0.0
        self.last_hp = None
