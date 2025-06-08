import threading
import time

from models.action import Action


class ThinkerThread(threading.Thread):
    """Analyzes states and makes decisions"""

    def __init__(self, agent, state_buffer, action_queue, perceptor_lock, decision_fps=60):
        super().__init__(daemon=True)
        self.agent = agent
        self.state_buffer = state_buffer
        self.action_queue = action_queue
        self.perceptor_lock = perceptor_lock
        self.decision_interval = 1.0 / decision_fps
        self.running = True
        self.last_state = None
        self.episode_memory = []
        self.step_count = 0

    def run(self):
        while self.running:
            start_time = time.time()

            try:
                # Get latest state
                with self.perceptor_lock:
                    current_state = self.state_buffer[-1] if self.state_buffer else None

                if current_state and current_state.screenshot is not None:
                    self.step_count += 1

                    # Make decision
                    action, duration, q_values = self.agent.get_action(
                        current_state,
                        episode_step_count=self.step_count
                    )

                    # Create action command
                    action_cmd = Action(action_type=action, duration=duration, frame_id=current_state.frame_id)
                    self.action_queue.put(action_cmd)

                    # Store for training
                    if self.last_state:
                        self.episode_memory.append({
                            'state': self.last_state,
                            'action': action_cmd,
                            'next_state': current_state
                        })

                    self.last_state = current_state

            except Exception as e:
                print(f"Thinker error: {e}")

            # Maintain decision rate
            elapsed = time.time() - start_time
            sleep_time = self.decision_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_episode_data(self):
        data = self.episode_memory.copy()
        self.episode_memory.clear()
        self.step_count = 0
        return data

    def stop(self):
        self.running = False
