from models.game_state import GameState


class RewardCalculator:
    """Reward system for Downwell.AI"""

    def __init__(self):
        # BASE REWARD
        self.base_survival = 0.001  # Minimal base reward for survival time

        # BIGGEST REWARDS/PENALTIES
        self.level_complete_bonus = 100.0  # Biggest positive reward
        self.death_penalty = -50.0         # Biggest negative penalty

        # HEALTH
        self.health_loss_penalty = -8.0  # Per HP lost - severe penalty
        self.health_gain_reward = 12.0   # Per HP gained - major reward

        # COMBO
        self.combo_base_reward = 0.5     # Per combo point maintained
        self.combo_growth_reward = 2.0   # Per combo point gained
        self.combo_break_penalty = -3.0  # When combo breaks from 3+
        self.high_combo_bonus = 1.0      # Extra reward for combo 5+

        # GEMS
        self.gem_base_reward = 1.0      # Per gem collected
        self.gem_high_multiplier = 2.0  # Multiplier during gem high
        self.gem_cap = 5.0              # Max gem reward per step

        # PROGRESS
        self.progress_reward = 0.3      # Per unit downward movement
        self.progress_cap = 1.5         # Max progress reward per step
        self.backward_penalty = -0.5    # Penalty for going up significantly

        # STAGNATION
        self.stagnation_threshold = 120  # Steps before penalty kicks in
        self.stagnation_penalty = -0.05  # Per step of stagnation
        self.max_stagnation_penalty = -0.5

        # Tracking variables
        self.stagnation_steps = 0
        self.last_position = None
        self.position_history = []

    def calculate_reward(self, state: GameState, next_state: GameState) -> float:
        # Handle level transitions (HP = 999 sentinel)
        if state.hp == 999.0 or next_state.hp == 999.0:
            return self.base_survival

        reward = self.base_survival

        # =================================================================
        # 1. LEVEL COMPLETION - Biggest positive reward
        # =================================================================
        level_completed = self._detect_level_completion(state.ypos, next_state.ypos)
        if level_completed:
            reward += self.level_complete_bonus
            self.stagnation_steps = 0
            print(f"🎉 LEVEL COMPLETE! Reward: +{self.level_complete_bonus}")

        # =================================================================
        # 2. DEATH PENALTY - Biggest negative penalty
        # =================================================================
        if next_state.hp <= 0:
            # Base death penalty
            death_reward = self.death_penalty

            # Additional penalty based on how poorly they were doing
            poor_performance_penalty = 0
            if state.combo == 0:
                poor_performance_penalty -= 5.0  # No combo when dying
            if next_state.gems < 10:
                poor_performance_penalty -= 3.0  # Very few gems

            reward += death_reward + poor_performance_penalty
            print(f"💀 DEATH! Penalty: {death_reward + poor_performance_penalty}")
            return max(reward, -100.0)  # Cap maximum death penalty

        # =================================================================
        # 3. HEALTH CHANGES - Major impact
        # =================================================================
        hp_change = next_state.hp - state.hp

        if hp_change > 0:  # Health gained
            health_reward = hp_change * self.health_gain_reward
            reward += health_reward
            if hp_change >= 1:
                print(f"💚 HP GAINED! +{health_reward:.1f}")

        elif hp_change < 0:  # Health lost
            health_penalty = abs(hp_change) * self.health_loss_penalty
            reward += health_penalty  # health_loss_penalty is negative
            print(f"💔 HP LOST! {health_penalty:.1f}")

        # =================================================================
        # 4. COMBO SYSTEM - High skill rewards
        # =================================================================
        combo_change = next_state.combo - state.combo

        # Base combo maintenance reward
        if next_state.combo > 0:
            combo_maintain_reward = next_state.combo * self.combo_base_reward
            reward += combo_maintain_reward

        # Combo growth reward
        if combo_change > 0:
            combo_growth = combo_change * self.combo_growth_reward
            reward += combo_growth
            if combo_change >= 2:
                print(f"🔥 COMBO GROWTH! +{combo_growth:.1f}")

        # Combo break penalty
        elif state.combo >= 3 and next_state.combo == 0:
            combo_break = self.combo_break_penalty * (state.combo / 3.0)  # Worse for higher combos
            reward += combo_break
            print(f"💥 COMBO BREAK! {combo_break:.1f}")

        # High combo bonus
        if next_state.combo >= 5:
            high_combo = self.high_combo_bonus * (next_state.combo - 4)
            reward += high_combo

        # =================================================================
        # 5. GEM COLLECTION - Moderate rewards
        # =================================================================
        gems_gained = max(0, next_state.gems - state.gems)
        if gems_gained > 0:
            gem_reward = min(gems_gained * self.gem_base_reward, self.gem_cap)

            # Apply gem high multiplier
            if next_state.gem_high:
                gem_reward *= self.gem_high_multiplier

            reward += gem_reward

        # =================================================================
        # 6. MOVEMENT AND PROGRESS
        # =================================================================
        y_progress = state.ypos - next_state.ypos  # Positive = downward movement

        if y_progress > 0.5:  # Meaningful downward progress
            progress_reward = min(y_progress * self.progress_reward, self.progress_cap)
            reward += progress_reward
            self.stagnation_steps = 0
        elif y_progress < -5.0:  # Significant upward movement (bad)
            reward += self.backward_penalty
        else:
            # Track stagnation
            self.stagnation_steps += 1

        # =================================================================
        # 7. STAGNATION PENALTY
        # =================================================================
        if self.stagnation_steps > self.stagnation_threshold:
            stagnation_penalty = min(
                (self.stagnation_steps - self.stagnation_threshold) * self.stagnation_penalty,
                self.max_stagnation_penalty
            )
            reward += stagnation_penalty

        # =================================================================
        # 8. FINAL REWARD CAPPING
        # =================================================================
        # Cap rewards to prevent extreme values (except level completion and death)
        if not level_completed and next_state.hp > 0:
            reward = max(-25.0, min(reward, 30.0))

        return reward

    @staticmethod
    def _detect_level_completion(old_y: float, new_y: float) -> bool:
        y_difference = new_y - old_y
        return y_difference > 1000 and old_y < -500 and new_y > -300

    def reset_episode(self):
        self.stagnation_steps = 0
        self.last_position = None
        self.position_history = []
