import glob
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

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
        logger.error(f"No data files found in {data_directory}")
        return None

    logger.info(f"Found {len(filepaths)} data files")

    for filepath in filepaths:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            all_data.extend(data)

    logger.info(f"Loaded {len(all_data)} frames total")
    return all_data


def train_on_batch_imitation(agent, batch):
    try:
        # Prepare batch tensors
        states = []
        actions = []

        for item in batch:
            visual_state = item["visual_state"]
            action = item["action"]

            # Skip invalid states
            if visual_state is None or visual_state.shape != (84, 84, 4):
                continue

            states.append(visual_state)
            actions.append(action)

        if len(states) < 2:  # Need at least 2 samples for batch norm
            return None

        # Convert to tensors
        state_tensor = torch.from_numpy(np.array(states)).float()
        action_tensor = torch.tensor(actions, dtype=torch.long)

        # Move to device and reshape for CNN: (B, H, W, C) -> (B, C, H, W)
        state_tensor = state_tensor.permute(0, 3, 1, 2).to(agent.device)
        action_tensor = action_tensor.to(agent.device)

        # Forward pass
        q_values = agent.q_network(state_tensor)

        # Cross-entropy loss for imitation learning
        # We want the network to predict the expert's actions
        loss = nn.CrossEntropyLoss()(q_values, action_tensor)

        # Backward pass
        agent.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(agent.q_network.parameters(), 10.0)

        agent.optimizer.step()

        return loss.item()

    except Exception as e:
        logger.error(f"Error in batch training: {e}")
        return None


def validate_data(dataset):
    valid_data = []
    invalid_count = 0

    for item in dataset:
        if "visual_state" in item and "action" in item:
            visual = item["visual_state"]
            if visual is not None and visual.shape == (84, 84, 4):
                valid_data.append(item)
            else:
                invalid_count += 1
        else:
            invalid_count += 1

    if invalid_count > 0:
        logger.warning(f"Filtered out {invalid_count} invalid samples")

    logger.info(f"Valid samples: {len(valid_data)}")
    return valid_data


def main():
    logger.remove()
    logger.add("pretrain_{time}.log", rotation="500 MB", level="DEBUG")
    logger.add(lambda msg: print(msg, end=""), level="INFO", colorize=True)

    logger.info("Starting pre-training")
    logger.info("=" * 50)

    config = AppConfig()

    # Load recorded gameplay data
    dataset = load_all_data(DATA_DIR)
    if not dataset:
        return

    # Validate and clean data
    dataset = validate_data(dataset)
    if len(dataset) < BATCH_SIZE:
        logger.error(f"Not enough valid data. Need at least {BATCH_SIZE} samples")
        return

    # Initialize the Agent
    agent = DQNAgent(
        action_space={i: set() for i in range(6)},
        config=config.agent,
        env_config=config.env,
        train_config=config.training,
    )

    # Override learning rate for pre-training
    for param_group in agent.optimizer.param_groups:
        param_group["lr"] = LEARNING_RATE

    # Pre-training loop
    logger.info(f"Starting pre-training for {EPOCHS} epochs")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Learning rate: {LEARNING_RATE}")

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{EPOCHS}")

        # Shuffle for each epoch
        random.shuffle(dataset)

        epoch_loss = 0.0
        num_batches = 0

        # Train in batches
        for i in range(0, len(dataset) - BATCH_SIZE, BATCH_SIZE):
            batch = dataset[i : i + BATCH_SIZE]

            loss = train_on_batch_imitation(agent, batch)

            if loss is not None:
                epoch_loss += loss
                num_batches += 1

                if num_batches % 100 == 0:
                    avg_batch_loss = epoch_loss / num_batches
                    logger.info(f"Batch {num_batches}: Avg Loss = {avg_batch_loss:.4f}")

        # Epoch statistics
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch + 1} complete. Average Loss: {avg_loss:.4f}")

            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                agent.save_model(MODEL_SAVE_PATH.replace(".pth", "_best.pth"))
                logger.success(f"New best model! Loss: {best_loss:.4f}")

        # Periodic save
        if (epoch + 1) % 10 == 0:
            agent.save_model(MODEL_SAVE_PATH)
            logger.info(f"Checkpoint saved to {MODEL_SAVE_PATH}")

    # Final save
    agent.save_model(MODEL_SAVE_PATH)
    logger.success(f"Pre-training complete! Final model saved to {MODEL_SAVE_PATH}")
    logger.info(f"Best loss achieved: {best_loss:.4f}")


if __name__ == "__main__":
    main()
