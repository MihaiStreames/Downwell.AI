import os
import pickle
import time
import uuid

import keyboard
import numpy as np
import pymem

from config import AppConfig
from environment.game_env import CustomDownwellEnvironment
from environment.mem_extractor import Player
from main import get_game_module

FPS = 60
FRAME_INTERVAL = 1.0 / FPS
DATA_DIR = "../gameplay_data"
os.makedirs(DATA_DIR, exist_ok=True)

ACTION_MAP = {
    frozenset(): 0,
    frozenset({"space"}): 1,
    frozenset({"left"}): 2,
    frozenset({"right"}): 3,
    frozenset({"left", "space"}): 4,
    frozenset({"right", "space"}): 5,
}


def get_current_action():
    pressed_keys = set()
    if keyboard.is_pressed("left"):
        pressed_keys.add("left")
    if keyboard.is_pressed("right"):
        pressed_keys.add("right")
    if keyboard.is_pressed("space"):
        pressed_keys.add("space")
    return ACTION_MAP.get(frozenset(pressed_keys), 0)


def main():
    print("Starting data recorder...")
    print("Switch to the Downwell window and start playing.")
    print("Press 'q' to stop recording.")

    config = AppConfig()

    try:
        proc = pymem.Pymem("downwell.exe")
        game_module = get_game_module(proc, "downwell.exe")
    except Exception as e:
        print(f"Error connecting to game: {e}")
        return

    player = Player(proc, game_module)
    env = CustomDownwellEnvironment(config.env)

    gameplay_data = []
    session_id = uuid.uuid4()

    print("Waiting for Downwell window...")
    while not env.window_exists():
        time.sleep(1)

    print("Window found. Initializing frame stack...")
    # Loop until we get a valid first frame to populate the stack
    while True:
        # We need to manually process the first few frames to fill the deque
        left, top, width, height = env.get_game_window_dimensions()
        import PIL.ImageGrab as ImageGrab

        screenshot = ImageGrab.grab(bbox=(left, top, left + width, top + height))
        frame = np.array(screenshot, dtype=np.uint8)[:, :, :3]

        cropped_frame = env.crop_game_area(frame)
        processed_frame = env._preprocess_frame(cropped_frame)

        if processed_frame is not None:
            for _ in range(config.env.frame_stack):
                env.frame_stack.append(processed_frame)
            print("Frame stack initialized. Recording will start in 3 seconds...")
            time.sleep(3)
            break
        else:
            print("Waiting for a valid initial frame...")
            time.sleep(0.1)

    while True:
        start_time = time.time()

        if keyboard.is_pressed("q"):
            print("Stopping recorder...")
            break

        # 1. Capture State (Visuals + Memory)
        visual_state = env.get_state()

        hp = player.get_value("hp")
        xpos = player.get_value("xpos")
        ypos = player.get_value("ypos")

        if visual_state is None or hp is None or xpos is None or ypos is None:
            continue

        memory_features = np.array(
            [
                hp,
                player.get_value("gems") or 0,
                player.get_value("combo") or 0,
                xpos,
                ypos,
                player.get_value("ammo") or 0,
            ],
            dtype=np.float32,
        )

        # 2. Get Human Action
        action = get_current_action()

        # 3. Store the data point
        gameplay_data.append(
            {
                "visual_state": visual_state,
                "memory_features": memory_features,
                "action": action,
            }
        )

        if len(gameplay_data) % 1000 == 0:
            print(f"Collected {len(gameplay_data)} frames...")
            chunk_num = len(gameplay_data) // 1000
            filepath = os.path.join(
                DATA_DIR, f"session_{session_id}_chunk_{chunk_num}.pkl"
            )
            with open(filepath, "wb") as f:
                pickle.dump(gameplay_data, f)
            print(f"Saved chunk to {filepath}")

        elapsed = time.time() - start_time
        sleep_time = FRAME_INTERVAL - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Final save
    if gameplay_data:
        filepath = os.path.join(DATA_DIR, f"session_{session_id}_final.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(gameplay_data, f)
        print(f"Recording complete. Saved all data to {filepath}")


if __name__ == "__main__":
    main()
