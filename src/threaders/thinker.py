import threading
import time

from loguru import logger

from src.models.action import Action


class ThinkerThread(threading.Thread):
    """Analyzes states, makes decisions, and trains the agent.

    This thread consumes states from the perception buffer, calculates
    rewards, trains the neural network, and queues actions for execution.

    Parameters
    ----------
    agent : DQNAgent
        Deep Q-Network agent for action selection and training.
    reward_calc : RewardCalculator
        Reward calculator for evaluating transitions.
    state_buffer : deque
        Shared buffer of game states from perception thread.
    action_queue : queue.Queue
        Queue to send actions to the actor thread.
    perceptor_lock : threading.Lock
        Lock for thread-safe access to state buffer.
    decision_fps : int, optional
        Target decisions per second, by default 60.

    Attributes
    ----------
    agent : DQNAgent
        Decision-making agent.
    reward_calc : RewardCalculator
        Reward calculation system.
    state_buffer : deque
        Shared state buffer.
    action_queue : queue.Queue
        Action command queue.
    perceptor_lock : threading.Lock
        Lock for buffer access.
    decision_interval : float
        Time between decisions in seconds.
    running : bool
        Flag to control thread execution.
    last_state : GameState | None
        Previous state for transition tracking.
    last_action : Action | None
        Previous action for transition tracking.
    step_count : int
        Steps taken in current episode.
    episode_reward : float
        Cumulative reward for current episode.
    current_reward : float
        Most recent reward value.
    experiences_added : int
        Number of experiences added to replay buffer.
    """

    def __init__(
        self,
        agent,
        reward_calc,
        state_buffer,
        action_queue,
        perceptor_lock,
        decision_fps=60,
    ):
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
        self.step_count = 0
        self.episode_reward = 0.0
        self.current_reward = 0.0
        self.experiences_added = 0

    def run(self) -> None:
        """Main thread loop for decision-making and training.

        Continuously reads latest state, calculates rewards for transitions,
        trains the agent on experiences, and selects actions to queue.
        Skips training during transition states (menus).
        """
        while self.running:
            start_time = time.time()

            try:
                with self.perceptor_lock:
                    current_state = self.state_buffer[-1] if self.state_buffer else None

                if current_state and current_state.screenshot is not None:
                    self.step_count += 1
                    is_transition_state = current_state.hp == 999.0

                    if self.last_state is not None and self.last_action is not None:
                        # Calculate reward for the transition
                        reward = self.reward_calc.calculate_reward(self.last_state, current_state)
                        self.current_reward = reward
                        self.episode_reward += reward

                        if not is_transition_state and self.last_state.hp != 999.0:
                            done = current_state.hp is not None and current_state.hp <= 0

                            loss = self.agent.train(
                                self.last_state,
                                self.last_action.action_type,
                                reward,
                                current_state,
                                done,
                            )
                            self.experiences_added += 1

                            if loss is not None and self.step_count % 100 == 0:
                                logger.debug(
                                    f"Step {self.step_count}: Loss = {loss:.4f}, Reward = {reward:.2f}"
                                )

                    # Make decision
                    action, q_values = self.agent.get_action(current_state)
                    action_cmd = Action(action_type=action, frame_id=current_state.frame_id)
                    self.action_queue.put(action_cmd)

                    # Update tracking
                    self.last_state = current_state
                    self.last_action = action_cmd

            except Exception as e:
                logger.error(f"Thinker error: {e}")

            # Maintain decision rate
            elapsed = time.time() - start_time
            sleep_time = self.decision_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_episode_stats(self) -> dict:
        """Get episode statistics and reset for next episode.

        Returns
        -------
        dict
            Dictionary containing episode_reward, experiences_added, and steps.

        Notes
        -----
        This method resets all episode tracking variables after returning stats.
        """
        stats = {
            "episode_reward": self.episode_reward,
            "experiences_added": self.experiences_added,
            "steps": self.step_count,
        }

        # Reset for next episode
        self.episode_reward = 0.0
        self.experiences_added = 0
        self.step_count = 0
        self.last_state = None
        self.last_action = None

        return stats

    def stop(self) -> None:
        """Stop the thinker thread gracefully."""
        self.running = False
