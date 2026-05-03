import sys


if sys.platform != "win32":
    raise RuntimeError("This AI only works on Windows.")
import csv
from pathlib import Path
import time

from loguru import logger
import pymem
from pymem.process import module_from_name

from src.agents.dqn_agent import DQNAgent
from src.config import Config
from src.core.orchestrator import DownwellAI
from src.core.reward_calculator import RewardCalculator
from src.core.vision import AIVision
from src.environment.game_env import CustomDownwellEnvironment
from src.environment.mem_extractor import Player


def main() -> None:
    logger.remove()
    logger.add("training_{time}.log", rotation="500 MB", level="DEBUG")
    logger.add(lambda msg: print(msg, end=""), level="DEBUG", colorize=True)
    logger.info("Starting Downwell.AI")

    Path("models").mkdir(exist_ok=True)
    config = Config()

    try:
        proc = pymem.Pymem("downwell.exe")
        game_module = module_from_name(proc.process_handle, "downwell.exe").lpBaseOfDll
    except Exception as e:
        logger.error(f"Failed to connect to game: {e}")
        return

    player = Player(proc, game_module)
    env = CustomDownwellEnvironment(config=config)
    agent = DQNAgent(action_space=env.actions, config=config)
    reward_calc = RewardCalculator(config=config)
    ai_system = DownwellAI(player, env, agent, reward_calc, config=config)
    vision = AIVision()

    best_reward = float("-inf")
    training_history = []
    episode = 0

    try:
        while episode < config.max_episodes:
            episode += 1
            logger.info(f"Episode {episode}")

            if env.reset(player) is None:
                logger.warning("Failed to reset, skipping episode")
                continue

            ai_system.start()
            episode_start = time.time()
            max_combo = 0
            final_gems = 0.0

            while True:
                state = ai_system.get_latest_state()
                if state:
                    _, q_values = agent.get_action(state)
                    last_reward = ai_system.thinker.current_reward if ai_system.thinker else 0.0
                    vision.display(state, q_values, last_reward)
                    max_combo = max(max_combo, state.combo)
                    final_gems = state.gems

                    if state.hp <= 0:
                        logger.info("Episode ended")
                        time.sleep(0.3)
                        break

            ai_system.stop()
            stats = ai_system.get_episode_stats()
            stats.update(
                {
                    "duration": time.time() - episode_start,
                    "max_combo": max_combo,
                    "final_gems": final_gems,
                    "epsilon": agent.epsilon,
                    "learning_rate": agent.scheduler.get_last_lr()[0],
                }
            )

            logger.info(
                f"Episode {episode} | reward={stats['episode_reward']:.1f}"
                f" dur={stats['duration']:.1f}s steps={stats['steps']}"
                f" combo={stats['max_combo']:.0f} gems={stats['final_gems']:.0f}"
                f" ypos={stats['max_ypos_reached']:.0f}"
                f" mem={len(agent.memory)}/{agent.memory.capacity}"
                f" ε={stats['epsilon']:.4f} lr={stats['learning_rate']:.6f}"
            )

            training_history.append(
                {
                    "episode": episode,
                    "reward": stats["episode_reward"],
                    "duration": stats["duration"],
                    "steps": stats["steps"],
                    "max_combo": stats["max_combo"],
                    "final_gems": stats["final_gems"],
                    "max_ypos_reached": stats["max_ypos_reached"],
                    "epsilon": stats["epsilon"],
                    "learning_rate": stats["learning_rate"],
                }
            )

            if stats["episode_reward"] > best_reward:
                best_reward = stats["episode_reward"]
                logger.success(f"NEW BEST: {best_reward:.2f}")
                agent.save_model("models/downwell_ai_best.pth")

            if episode % config.save_frequency == 0:
                agent.save_model(f"models/downwell_ai_{episode}.pth")
                logger.info(f"Checkpoint saved: episode {episode}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted")
    except Exception as e:
        logger.exception(f"Training error: {e}")
    finally:
        ai_system.stop()
        vision.close()

        if training_history:
            with Path("training_history.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, training_history[0].keys())
                writer.writeheader()
                writer.writerows(training_history)
            logger.success("Saved training_history.csv")

        try:
            agent.save_model(f"models/downwell_ai_final_{episode}.pth")
            logger.success(f"Final model saved (episode {episode}, best={best_reward:.2f})")
        except Exception as e:
            logger.error(f"Error saving final model: {e}")


if __name__ == "__main__":
    main()
