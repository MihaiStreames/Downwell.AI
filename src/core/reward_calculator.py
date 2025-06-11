from config import RewardConfig
from models.game_state import GameState


class RewardCalculator:
    """Reward system for Downwell.AI"""

    def __init__(self, config: RewardConfig):
        self.config = config
        self.stagnation_steps = 0
        self.max_depth_achieved = 0.0

    @staticmethod
    def _detect_level_completion(state: GameState, next_state: GameState) -> bool:
        if state is None or state.xpos is None or state.ypos is None: return False
        player_was_in_well = state.ypos < -100
        player_is_in_menu = next_state.xpos is None or next_state.hp is None
        return player_was_in_well and player_is_in_menu

    def calculate_reward(self, state: GameState, next_state: GameState) -> float:
        if self._detect_level_completion(state, next_state):
            print(f"🎉 LEVEL COMPLETE! Reward: +{self.config.level_complete_bonus}")
            self.reset_episode()
            return self.config.level_complete_bonus

        if state.hp == 999.0 or next_state.hp == 999.0: return 0.0

        if state.hp > 0 and (next_state.hp is None or next_state.hp <= 0):
            return self.config.death_penalty

        if next_state.hp is None or next_state.hp <= 0: return 0.0

        reward = self.config.base_survival

        # Health changes
        hp_change = next_state.hp - state.hp
        if hp_change > 0:
            reward += hp_change * self.config.health_gain_reward
        elif hp_change < 0:
            reward += abs(hp_change) * self.config.health_loss_penalty

        # Gem collection
        gems_gained = max(0, next_state.gems - state.gems)
        if gems_gained > 0: reward += gems_gained * self.config.gem_base_reward

        # Combo growth
        combo_change = next_state.combo - state.combo
        if combo_change > 0: reward += combo_change * self.config.combo_growth_reward

        # Movement and Stagnation
        if state.ypos is not None and next_state.ypos is not None:
            # 1. Reward for reaching a new depth
            if next_state.ypos < self.max_depth_achieved:
                progress = self.max_depth_achieved - next_state.ypos
                reward += progress * self.config.new_depth_reward
                self.max_depth_achieved = next_state.ypos

            # 2. Stagnation and Backward penalty
            y_diff = state.ypos - next_state.ypos
            x_diff = abs(state.xpos - next_state.xpos) if state.xpos and next_state.xpos else 0

            if y_diff < -2.0:  # Moving up is penalized
                reward += self.config.backward_penalty

            if abs(y_diff) < 1.0 and x_diff < 1.0:  # If not moving much on either axis
                self.stagnation_steps += 1
            else:
                self.stagnation_steps = 0  # Reset if moving

        if self.stagnation_steps > self.config.stagnation_threshold:
            reward += self.config.stagnation_penalty

        # Clip the reward to prevent extreme values
        return max(self.config.min_reward_clip, min(reward, self.config.max_reward_clip))

    def reset_episode(self):
        self.stagnation_steps = 0
        self.max_depth_achieved = 0.0
