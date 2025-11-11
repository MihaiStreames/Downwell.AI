import csv
import os
import platform
import time
from typing import Any, Dict, List, Optional

import pymem
from loguru import logger
from pymem.process import module_from_name

from agents.dqn_agent import DQNAgent
from config import AppConfig
from core.orchestrator import DownwellAI
from core.reward_calculator import RewardCalculator
from core.vision import AIVision
from environment.game_env import CustomDownwellEnvironment
from environment.mem_extractor import Player


def setup_logging() -> None:
    """Configure loguru logging with file and console outputs."""
    logger.remove()
    logger.add("training_{time}.log", rotation="500 MB", level="DEBUG")
    logger.add(lambda msg: print(msg, end=""), level="DEBUG", colorize=True)

    logger.info("Starting Downwell.AI")
    logger.info("=" * 50)


def check_platform() -> bool:
    """Check if the platform is Windows."""
    if platform.system() != "Windows":
        logger.critical("ERROR: This AI only works on Windows.")
        return False
    return True


def get_game_module(proc: pymem.Pymem, executable_name: str) -> int:
    """Get the base address of the game module."""
    try:
        return module_from_name(proc.process_handle, executable_name).lpBaseOfDll
    except Exception as e:
        logger.error(f"Error finding module '{executable_name}': {str(e)}")
        raise


def connect_to_game(executable_name: str) -> Optional[tuple[pymem.Pymem, int]]:
    """Connect to the game process."""
    logger.info(f"Connecting to {executable_name}...")

    try:
        proc = pymem.Pymem(executable_name)
        game_module = get_game_module(proc, executable_name)
        logger.success("Successfully connected to game")
        return proc, game_module
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        return None


def initialize_components(
    proc: pymem.Pymem, game_module: int, config: AppConfig
) -> Optional[Dict[str, Any]]:
    """Initialize all AI components."""
    try:
        player = Player(proc, game_module)
        env = CustomDownwellEnvironment(config=config.env)

        agent = DQNAgent(
            action_space=env.actions,
            config=config.agent,
            env_config=config.env,
            train_config=config.training,
        )

        reward_calc = RewardCalculator(config=config.rewards)
        ai_system = DownwellAI(player, env, agent, reward_calc, config=config.env)
        vision = AIVision()

        return {
            "player": player,
            "env": env,
            "agent": agent,
            "reward_calc": reward_calc,
            "ai_system": ai_system,
            "vision": vision,
        }

    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        return None


def run_episode(
    episode_num: int, components: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Run a single training episode."""
    logger.info(f"Episode {episode_num}")

    env = components["env"]
    player = components["player"]
    ai_system = components["ai_system"]
    agent = components["agent"]
    vision = components["vision"]

    # Reset environment
    state = env.reset(player)
    if state is None:
        logger.warning("Failed to reset, skipping episode")
        return None

    # Start AI system
    ai_system.start()

    # Monitor episode
    episode_start = time.time()
    max_combo = 0

    while True:
        current_state = ai_system.get_latest_state()

        if current_state:
            # Update AI vision
            _, q_values = agent.get_action(current_state)
            last_reward = ai_system.thinker.current_reward if ai_system.thinker else 0.0
            vision.display(current_state, q_values, last_reward)

            max_combo = max(max_combo, current_state.combo)
            final_gems = current_state.gems

            if current_state.hp <= 0:
                logger.info("Episode ended")
                time.sleep(0.3)
                break

    # Stop AI system and get learning statistics
    ai_system.stop()
    episode_stats = ai_system.get_episode_stats()
    episode_duration = time.time() - episode_start

    # Add additional stats
    episode_stats.update(
        {
            "duration": episode_duration,
            "max_combo": max_combo,
            "final_gems": final_gems,
            "epsilon": agent.epsilon,
            "learning_rate": agent.scheduler.get_last_lr()[0],
        }
    )

    return episode_stats


def log_episode_summary(
    episode_num: int, stats: Dict[str, Any], agent: DQNAgent
) -> None:
    """Log episode summary information."""
    logger.info(f"Episode {episode_num} Summary:")
    logger.info(f"Reward: {stats['episode_reward']:.1f}")
    logger.info(f"Duration: {stats['duration']:.1f}s")
    logger.info(f"Steps: {stats['steps']}")
    logger.info(f"Max Combo: {stats['max_combo']:.0f}")
    logger.info(f"Final Gems: {stats['final_gems']:.0f}")
    logger.info(f"Experiences: +{stats['experiences_added']} (total: {len(agent.memory)}/{agent.memory.capacity})")
    logger.info(f"Epsilon: {stats['epsilon']:.4f}")
    logger.info(f"Learning Rate: {stats['learning_rate']:.6f}")


def save_episode_data(episode_num: int, stats: Dict[str, Any]) -> Dict[str, Any]:
    """Create episode data dictionary for history tracking."""
    return {
        "episode": episode_num,
        "reward": stats["episode_reward"],
        "duration": stats["duration"],
        "steps": stats["steps"],
        "max_combo": stats["max_combo"],
        "final_gems": stats["final_gems"],
        "epsilon": stats["epsilon"],
        "learning_rate": stats["learning_rate"],
    }


def handle_checkpoints(
    episode_num: int,
    episode_reward: float,
    best_reward: float,
    agent: DQNAgent,
    config: AppConfig,
) -> float:
    """Handle model checkpointing and updates."""
    # Check for new best
    if episode_reward > best_reward:
        best_reward = episode_reward
        logger.success(f"NEW BEST: {best_reward:.2f}")
        agent.save_model("models/downwell_ai_best.pth")

    # Periodic save
    if episode_num % config.training.save_frequency == 0:
        model_path = f"models/downwell_ai_{episode_num}.pth"
        agent.save_model(model_path)
        logger.info(f"Model saved: {model_path}")

    # Target network update
    if episode_num % config.training.target_update_frequency == 0:
        agent.update_target_network()
        logger.info(" Target network updated")

    return best_reward


def save_training_history(training_history: List[Dict[str, Any]]) -> None:
    """Save training history to CSV file."""
    if not training_history:
        return

    logger.info("Saving training history...")
    keys = training_history[0].keys()
    with open("training_history.csv", "w", newline="") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(training_history)
    logger.success("Saved training history to training_history.csv")


def cleanup(
    components: Optional[Dict[str, Any]],
    training_history: List[Dict[str, Any]],
    episode_num: int,
    best_reward: float,
) -> None:
    """Clean up resources and save final state."""
    logger.info("Cleaning up...")

    # Stop AI system and close vision
    if components:
        if "ai_system" in components:
            components["ai_system"].stop()
        if "vision" in components:
            components["vision"].close()

    # Save training history
    save_training_history(training_history)

    # Save final model
    if components and "agent" in components:
        try:
            final_model_path = f"models/downwell_ai_final_{episode_num}.pth"
            components["agent"].save_model(final_model_path)
            logger.success(f"Final model saved: {final_model_path}")
        except Exception as e:
            logger.error(f"Error saving final model: {e}")

    logger.info("Training session ended")
    logger.info(f"Total episodes completed: {episode_num}")
    logger.info(f"Best episode reward: {best_reward:.2f}")


def training_loop(components: Dict[str, Any], config: AppConfig) -> None:
    """Main training loop."""
    max_episodes = config.training.max_episodes
    episode = 0
    best_reward = float("-inf")
    training_history = []

    try:
        while episode < max_episodes:
            episode += 1

            # Run episode
            episode_stats = run_episode(episode, components)

            if episode_stats is None:
                continue

            # Log summary
            log_episode_summary(episode, episode_stats, components["agent"])

            # Save episode data
            episode_data = save_episode_data(episode, episode_stats)
            training_history.append(episode_data)

            # Handle checkpoints
            best_reward = handle_checkpoints(
                episode,
                episode_stats["episode_reward"],
                best_reward,
                components["agent"],
                config,
            )

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.exception(f"Training error: {e}")
    finally:
        cleanup(components, training_history, episode, best_reward)


def main() -> None:
    """Main entry point for the training script."""
    # Setup
    setup_logging()
    os.makedirs("models", exist_ok=True)

    # Check platform
    if not check_platform():
        return

    # Load configuration
    config = AppConfig()

    # Connect to game
    executable_name = "downwell.exe"
    connection = connect_to_game(executable_name)
    if connection is None:
        return

    proc, game_module = connection

    # Initialize components
    components = initialize_components(proc, game_module, config)
    if components is None:
        return

    # Run training
    training_loop(components, config)


if __name__ == "__main__":
    main()
