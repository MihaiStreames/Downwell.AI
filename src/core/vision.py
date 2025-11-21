import cv2
import numpy as np


class AIVision:
    """Window that displays AI stats and the agent's Q-values"""

    def __init__(self, width=300, height=250):
        self.width = width
        self.height = height
        self.window_name = "AI Vision"

        # Action names corresponding to the Q-value indices
        self.action_names = ["none", "jump", "left", "right", "left+jump", "right+jump"]

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.window_name, 100, 100)
        cv2.resizeWindow(self.window_name, self.width, self.height)

    def display(self, game_state, q_values, last_reward=0.0):
        # Create a black canvas
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Game State Info
        if game_state:
            hp_text = f"HP: {game_state.hp:.0f}" if game_state.hp is not None else "HP: N/A"
            gems_text = (
                f"Gems: {game_state.gems:.0f}" if game_state.gems is not None else "Gems: N/A"
            )
            combo_text = (
                f"Combo: {game_state.combo:.0f}" if game_state.combo is not None else "Combo: N/A"
            )
            ammo_text = (
                f"Ammo: {game_state.ammo:.0f}" if game_state.ammo is not None else "Ammo: N/A"
            )
            xpos_text = (
                f"X Pos: {game_state.xpos:.0f}" if game_state.xpos is not None else "X Pos: N/A"
            )
            ypos_text = (
                f"Y Pos: {game_state.ypos:.0f}" if game_state.ypos is not None else "Y Pos: N/A"
            )
            gem_high_text = (
                f"Gem High: {game_state.gem_high:.0f}"
                if game_state.gem_high is not None
                else "Gem High: N/A"
            )
            reward_text = f"Reward: {last_reward:.2f}"

            stats = [
                hp_text,
                gems_text,
                combo_text,
                ammo_text,
                xpos_text,
                ypos_text,
                gem_high_text,
                reward_text,
            ]
            for i, text in enumerate(stats):
                cv2.putText(
                    canvas,
                    text,
                    (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

        # Q-Value Bars
        if q_values is not None and len(q_values) > 0:
            # Normalize Q-values for consistent bar height
            max_q = max(q_values) if max(q_values) > 0 else 1.0
            min_q = min(q_values)
            q_range = max_q - min_q if max_q != min_q else 1.0

            bar_x_start = 150  # X-coordinate to start drawing bars

            for i, q_val in enumerate(q_values):
                # Bar color: Green for the best action, gray for others
                color = (0, 255, 0) if i == np.argmax(q_values) else (100, 100, 100)

                # Calculate bar width based on normalized Q-value
                normalized_q = (q_val - min_q) / q_range if q_range > 0 else 0.5
                bar_width = int(normalized_q * (self.width - bar_x_start - 10))
                bar_width = max(1, bar_width)  # Ensure bar is at least 1px wide

                y_pos = 25 + i * 22
                # Draw the bar
                cv2.rectangle(
                    canvas,
                    (bar_x_start, y_pos - 12),
                    (bar_x_start + bar_width, y_pos),
                    color,
                    -1,
                )
                # Draw the text label
                text = f"{self.action_names[i]}: {q_val:.2f}"
                cv2.putText(
                    canvas,
                    text,
                    (bar_x_start, y_pos - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )

        # Show the canvas
        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyWindow(self.window_name)
