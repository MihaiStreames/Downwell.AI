import threading
import time

from models.game_state import GameState


class PerceptorThread(threading.Thread):
    """Continuously captures game state at high frequency"""

    def __init__(self, player, env, state_buffer, target_fps=60):
        super().__init__(daemon=True)
        self.player = player
        self.env = env
        self.state_buffer = state_buffer
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.running = True
        self.frame_count = 0
        self.lock = threading.Lock()

    def run(self):
        while self.running:
            start_time = time.time()
            try:
                # Capture game state
                screenshot = self.env.get_state()

                # Get game state values with HP sentinel only
                hp = self.player.get_value('hp')
                if hp is None: hp = 999.0  # Sentinel for level transitions

                # Get game state
                gems = self.player.get_value('gems') or 0
                combo = self.player.get_value('combo') or 0
                xpos = self.player.get_value('xpos') or 0
                ypos = self.player.get_value('ypos') or 0
                ammo = self.player.get_value('ammo') or 0
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
                    frame_id=self.frame_count
                )

                with self.lock:
                    self.state_buffer.append(state)
                    self.frame_count += 1

            except Exception as e:
                print(f"Perceptor error: {e}")

            elapsed = time.time() - start_time
            sleep_time = self.frame_interval - elapsed
            if sleep_time > 0: time.sleep(sleep_time)

    def stop(self):
        self.running = False
