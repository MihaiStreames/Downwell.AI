import threading
import time

from loguru import logger

from models.game_state import GameState


class PerceptorThread(threading.Thread):
    """Continuously captures game state at high frequency"""

    def __init__(self, player, env, state_buffer, perception_fps=60):
        super().__init__(daemon=True)
        self.player = player
        self.env = env
        self.state_buffer = state_buffer
        self.target_fps = perception_fps
        self.frame_interval = 1.0 / perception_fps
        self.running = True
        self.frame_count = 0
        self.lock = threading.Lock()

    def run(self):
        while self.running:
            start_time = time.time()
            try:
                # Capture game state
                screenshot = self.env.get_state()

                # Get game state values
                hp = self.player.get_value("hp")
                xpos = self.player.get_value("xpos")
                ypos = self.player.get_value("ypos")

                # Check if we are in a transition state
                is_transition_state = xpos is None or hp is None

                # If in a transition, set a sentinel value for HP for other parts of the system,
                # but keep xpos/ypos as None to be used for level detection.
                if is_transition_state:
                    hp = 999.0

                gems = self.player.get_value("gems") or 0
                combo = self.player.get_value("combo") or 0
                ammo = self.player.get_value("ammo") or 0
                gem_high = self.player.is_gem_high()

                state = GameState(
                    screenshot=screenshot,
                    hp=hp,
                    gems=gems,
                    combo=combo,
                    xpos=xpos,
                    ypos=ypos,
                    ammo=ammo,
                    gem_high=gem_high,
                    timestamp=time.time(),
                    frame_id=self.frame_count,
                )

                with self.lock:
                    self.state_buffer.append(state)
                    self.frame_count += 1

            except Exception as e:
                logger.error(f"Perceptor error: {e}")

            elapsed = time.time() - start_time
            sleep_time = self.frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self):
        self.running = False
