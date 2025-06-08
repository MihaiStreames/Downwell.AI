import threading
import time

from models.action import Action


class ThinkerThread(threading.Thread):
    """Analyzes states and makes decisions"""

    def __init__(self, agent, reward_calc, state_buffer, action_queue, perceptor_lock, decision_fps=60):
        super().__init__(daemon=True)
        self.agent = agent
        self.reward_calc = reward_calc
        self.state_buffer = state_buffer
        self.action_queue = action_queue
        self.perceptor_lock = perceptor_lock
        self.decision_interval = 1.0 / decision_fps
        self.running = True

        # Learning state tracking
        self.last_state = None
        self.last_action = None
        self.last_memory_features = None
        self.step_count = 0
        self.episode_reward = 0.0
        self.experiences_added = 0

    def run(self):
        while self.running:
            start_time = time.time()

            try:
                # Get latest state
                with self.perceptor_lock:
                    current_state = self.state_buffer[-1] if self.state_buffer else None

                if current_state and current_state.screenshot is not None:
                    self.step_count += 1

                    # Extract memory features for current state
                    current_memory_features = self.agent.extract_memory_features(current_state)

                    # LEARNING STEP: If we have a previous state, learn from the transition
                    if self.last_state is not None and self.last_action is not None:
                        # Calculate reward for the transition
                        reward = self.reward_calc.calculate_reward(self.last_state, current_state)
                        self.episode_reward += reward

                        # Check if episode is done (dead, but not during level transitions)
                        done = current_state.hp <= 0 and current_state.hp != 999.0

                        # Train
                        loss = self.agent.train(
                            self.last_state,
                            self.last_action.action_type,
                            self.last_action.duration,
                            reward,
                            current_state,
                            done,
                            self.last_memory_features,
                            current_memory_features
                        )
                        self.experiences_added += 1

                        # Log progress periodically
                        if loss is not None and self.step_count % 100 == 0:
                            print(f"Step {self.step_count}: Loss = {loss:.4f}, Reward = {reward:.2f}")

                    # Make decision for current state
                    action, duration, q_values = self.agent.get_action(current_state)

                    # Create action command
                    action_cmd = Action(action_type=action, duration=duration, frame_id=current_state.frame_id)
                    self.action_queue.put(action_cmd)

                    # Update tracking variables for next iteration
                    self.last_state = current_state
                    self.last_action = action_cmd
                    self.last_memory_features = current_memory_features

            except Exception as e:
                print(f"Thinker error: {e}")

            # Maintain decision rate
            elapsed = time.time() - start_time
            sleep_time = self.decision_interval - elapsed
            if sleep_time > 0: time.sleep(sleep_time)

    def get_episode_stats(self):
        stats = {
            'episode_reward': self.episode_reward,
            'experiences_added': self.experiences_added,
            'steps': self.step_count
        }

        # Reset for next episode
        self.episode_reward = 0.0
        self.experiences_added = 0
        self.step_count = 0
        self.last_state = None
        self.last_action = None
        self.last_memory_features = None

        return stats

    def stop(self):
        self.running = False
