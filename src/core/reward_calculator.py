from config import RewardConfig
from models.game_state import GameState


class RewardCalculator:
    """Reward system for Downwell.AI"""

    def __init__(self, config: RewardConfig):
        self.config = config

        # Tracking variables
        self.stagnation_steps = 0

    def calculate_reward(self, state: GameState, next_state: GameState) -> float:
        # =================================================================
        # 1. LEVEL COMPLETION - Biggest positive reward
        # =================================================================
        if self._detect_level_completion(state, next_state):
            print(f"🎉 LEVEL COMPLETE! Reward: +{self.config.level_complete_bonus}")
            return self.config.level_complete_bonus

        # Handle level transitions (HP = 999 sentinel)
        if state.hp == 999.0 or next_state.hp == 999.0: return self.config.base_survival
        reward = self.config.base_survival

        # =================================================================
        # 2. DEATH PENALTY - Biggest negative penalty
        # =================================================================
        if state.hp > 0 >= next_state.hp:
            # Base death penalty
            death_reward = self.config.death_penalty

            # Additional penalty based on how poorly they were doing
            poor_performance_penalty = 0
            if state.combo == 0: poor_performance_penalty -= 5.0  # No combo when dying
            if next_state.gems < 10: poor_performance_penalty -= 3.0  # Very few gems

            reward += death_reward + poor_performance_penalty
            print(f"💀 DEATH! Penalty: {death_reward + poor_performance_penalty}")
            return max(reward, -100.0)  # Cap maximum death penalty

        # If the agent is already dead, return a neutral reward and stop calculating
        if state.hp <= 0: return 0.0

        # =================================================================
        # 3. HEALTH CHANGES - Major impact
        # =================================================================
        hp_change = next_state.hp - state.hp

        if hp_change > 0:  # Health gained
            health_reward = hp_change * self.config.health_gain_reward
            reward += health_reward
            if hp_change >= 1: print(f"💚 HP GAINED! +{health_reward:.1f}")

        elif hp_change < 0:  # Health lost
            health_penalty = abs(hp_change) * self.config.health_loss_penalty
            reward += health_penalty
            print(f"💔 HP LOST! {health_penalty:.1f}")

        # =================================================================
        # 4. COMBO SYSTEM - High skill rewards
        # =================================================================
        combo_change = next_state.combo - state.combo

        # Base combo maintenance reward
        # if next_state.combo > 0:
        #     combo_maintain_reward = next_state.combo * self.config.combo_base_reward
        #     reward += combo_maintain_reward

        # Combo growth reward
        if combo_change > 0:
            combo_growth = combo_change * self.config.combo_growth_reward
            reward += combo_growth
            if combo_change >= 1: print(f"🔥 COMBO GROWTH! +{combo_growth:.1f}")

        # Combo break penalty
        # elif state.combo >= 3 and next_state.combo == 0:
        #     combo_break = self.config.combo_break_penalty * (state.combo / 3.0)  # Worse for higher combos
        #     reward += combo_break
        #     print(f"💥 COMBO BREAK! {combo_break:.1f}")

        # High combo bonus
        if next_state.combo >= 5:
            high_combo = self.config.high_combo_bonus * (next_state.combo - 4)
            reward += high_combo
            print(f"🌟 HIGH COMBO! +{high_combo:.1f}")

        # =================================================================
        # 5. GEM COLLECTION - Moderate rewards
        # =================================================================
        gems_gained = max(0, next_state.gems - state.gems)
        if gems_gained > 0:
            gem_reward = min(gems_gained * self.config.gem_base_reward, 5)

            # Apply gem high multiplier
            if next_state.gem_high: gem_reward *= self.config.gem_high_multiplier
            reward += gem_reward
            print(f"💎 GEMS COLLECTED! +{gem_reward:.1f}")

        # =================================================================
        # 6. MOVEMENT AND PROGRESS
        # =================================================================
        y_progress = state.ypos - next_state.ypos  # Positive = downward movement

        if y_progress > 0.5:  # Meaningful downward progress
            progress_reward = min(y_progress * self.config.progress_reward, 1.5)
            reward += progress_reward
            self.stagnation_steps = 0
        elif y_progress < -5.0:  # Significant upward movement (bad)
            reward += self.config.backward_penalty
        else:
            # Track stagnation
            self.stagnation_steps += 1

        # =================================================================
        # 7. STAGNATION PENALTY
        # =================================================================
        if self.stagnation_steps > self.config.stagnation_threshold:
            stagnation_penalty = min(
                (self.stagnation_steps - self.config.stagnation_threshold) * self.config.stagnation_penalty, -0.5)
            reward += stagnation_penalty

        # =================================================================
        # 8. FINAL REWARD CAPPING
        # =================================================================
        # Cap rewards to prevent extreme values
        if next_state.hp > 0: reward = max(-25.0, min(reward, 30.0))
        return reward

    @staticmethod
    def _detect_level_completion(state: GameState, next_state: GameState) -> bool:
        # Ensure we have a valid previous state to compare against
        if state is None or state.xpos is None or state.ypos is None: return False

        # Condition 1: The player was verifiably in the game world
        player_was_in_well = state.ypos < -300

        # Condition 2: The player's state is now unreadable
        player_is_in_menu = next_state.xpos is None or next_state.hp is None

        return player_was_in_well and player_is_in_menu

    def reset_episode(self):
        self.stagnation_steps = 0
