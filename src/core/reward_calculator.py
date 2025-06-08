from models.game_state import GameState


class RewardCalculator:
    """Reward system for Downwell.AI"""

    def __init__(self):
        self.base_survival = 0.05
        self.health_scale = 0.3
        self.damage_scale = 2.0
        self.health_gain_scale = 3.0
        self.gem_scale = 1.5
        self.gem_cap = 3.0
        self.combo_scale = 0.2
        self.combo_cap = 1.5
        self.progress_scale = 1.0
        self.progress_cap = 2.0
        self.death_base = 8.0
        self.gem_high_multiplier = 2.5
        self.level_complete_bonus = 15.0

        # Track for stagnation
        self.stagnation_steps = 0

    def calculate_reward(self, state: GameState, next_state: GameState) -> float:
        if state.hp == 999.0 or next_state.hp == 999.0:
            return self.base_survival

        reward = self.base_survival

        # 1. Health changes
        hp_change = next_state.hp - state.hp

        if hp_change > 0:  # Health gained
            health_gain_reward = min(hp_change * self.health_gain_scale, 5.0)
            reward += health_gain_reward
        elif hp_change < 0:  # Health lost
            damage_penalty = min(abs(hp_change) * self.damage_scale, 3.0)
            reward -= damage_penalty

        # Health ratio bonus
        health_ratio = max(0, next_state.hp) / 4.0
        reward += health_ratio * self.health_scale

        # 2. Gem collection with gem high bonus
        gems_gained = max(0, next_state.gems - state.gems)
        if gems_gained > 0:
            gem_reward = min(gems_gained * self.gem_scale, self.gem_cap)

            if next_state.gem_high:
                gem_reward *= self.gem_high_multiplier

            reward += gem_reward

        # 3. Combo rewards
        current_combo_bonus = min(next_state.combo * self.combo_scale, self.combo_cap)
        reward += current_combo_bonus

        combo_growth = max(0, next_state.combo - state.combo)
        if combo_growth > 0:
            reward += min(combo_growth * 0.5, 1.0)

        # Combo break penalty
        if state.combo >= 3 and next_state.combo == 0:
            reward -= min(state.combo * 0.2, 1.5)

        # 4. Progress
        y_progress = state.ypos - next_state.ypos

        # Detect level completion
        level_completed = self._detect_level_completion(state.ypos, next_state.ypos)
        if level_completed:
            reward += self.level_complete_bonus
            self.stagnation_steps = 0
        elif y_progress > 0.3:  # Any meaningful progress
            progress_reward = min(y_progress * self.progress_scale, self.progress_cap)
            reward += progress_reward
            self.stagnation_steps = 0
        elif y_progress < -3.0:  # Going up penalty
            reward -= 1.0
        else:
            self.stagnation_steps += 1

        # 5. Stagnation penalty
        if self.stagnation_steps > 20:
            reward -= min((self.stagnation_steps - 20) * 0.05, 0.5)

        # 6. Death penalty
        if next_state.hp <= 0:
            progress_made = max(0, abs(state.ypos) * 0.01)
            death_penalty = self.death_base + min(progress_made, 5.0)
            reward -= death_penalty

        return max(-15.0, min(reward, 25.0))

    @staticmethod
    def _detect_level_completion(old_y: float, new_y: float) -> bool:
        y_difference = new_y - old_y
        return y_difference > 1000 and old_y < -500 and new_y > -300

    def reset_episode(self):
        self.stagnation_steps = 0
