import glob
import os
import pickle
import random

from agents.dqn_agent import DQNAgent
from config import AppConfig

DATA_DIR = "../gameplay_data"
MODEL_SAVE_PATH = "../models/pretrained_agent.pth"
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.0005


def load_all_data(data_directory):
    all_data = []
    filepaths = glob.glob(os.path.join(data_directory, "*.pkl"))
    if not filepaths:
        print(f"No data files found in {data_directory}. Please record some gameplay first.")
        return None

    print(f"Found {len(filepaths)} data files. Loading...")
    for filepath in filepaths:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            all_data.extend(data)
    print(f"Loaded a total of {len(all_data)} frames.")
    return all_data


def main():
    config = AppConfig()

    # 1. Load recorded gameplay data
    dataset = load_all_data(DATA_DIR)
    if not dataset:
        return

    # 2. Initialize the Agent
    agent = DQNAgent(
        action_space={i: set() for i in range(6)},
        config=config.agent,
        env_config=config.env
    )

    for param_group in agent.optimizer.param_groups:
        param_group['lr'] = LEARNING_RATE

    # 3. The Pre-training Loop
    print(f"Starting pre-training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")

        # Shuffle the dataset for each epoch
        random.shuffle(dataset)

        epoch_loss = 0.0
        num_batches = 0

        # Iterate over the dataset in batches
        for i in range(0, len(dataset), BATCH_SIZE):
            batch = dataset[i:i + BATCH_SIZE]
            if len(batch) < BATCH_SIZE:
                continue

            # Use the new imitation learning method
            loss = agent.train_on_batch_imitation(batch)

            if loss is not None:
                epoch_loss += loss
                num_batches += 1

            if num_batches % 100 == 0 and num_batches > 0:
                print(f"  Batch {num_batches}: Average Loss = {epoch_loss / num_batches:.4f}")

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch + 1} complete. Average Loss: {avg_loss:.4f}")

        # Save the model after each epoch
        agent.save_model(MODEL_SAVE_PATH)
        print(f"Saved pre-trained model to {MODEL_SAVE_PATH}")

    print("\nPre-training finished!")


if __name__ == "__main__":
    main()
