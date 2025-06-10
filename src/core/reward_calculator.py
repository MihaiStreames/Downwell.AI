from collections import deque

from config import RewardConfig
from models.game_state import GameState


class RewardCalculator:
    """Reward system for Downwell.AI"""

    def __init__(self, config: RewardConfig):
        self.config = config
        self.stagnation_steps = 0
        self.y_history = deque(maxlen=120)

    @staticmethod
    def _detect_level_completion(state: GameState, next_state: GameState) -> bool:
        if state is None or state.xpos is None or state.ypos is None: return False
        player_was_in_well = state.ypos < -100
        player_is_in_menu = next_state.xpos is None or next_state.hp is None
        return player_was_in_well and player_is_in_menu

    def calculate_reward(self, state: GameState, next_state: GameState) -> float:
        if self._detect_level_completion(state, next_state):
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
        if state.ypos is not None: self.y_history.append(state.ypos)

        if state.ypos is not None and next_state.ypos is not None and len(self.y_history) > 0:
            y_progress = max(self.y_history) - next_state.ypos

            if y_progress > 2.0:
                reward += y_progress * self.config.progress_reward
                self.stagnation_steps = 0
            elif (state.ypos - next_state.ypos) < -2.0:
                reward += self.config.backward_penalty
                self.y_history.clear()
            else:
                if state.xpos is not None and next_state.xpos is not None:
                    x_diff = abs(state.xpos - next_state.xpos)
                    if x_diff > 1.0:
                        self.stagnation_steps = 0
                    else:
                        self.stagnation_steps += 1
                else:
                    self.stagnation_steps = 0
        else:
            self.stagnation_steps = 0

        if self.stagnation_steps > self.config.stagnation_threshold: reward += self.config.stagnation_penalty
        return reward

    def reset_episode(self):
        self.stagnation_steps = 0
        self.y_history.clear()
