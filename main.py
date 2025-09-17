import csv
import os
import platform
import time
from collections import deque

import pymem
from pymem.process import module_from_name

from agents.dqn_agent import DQNAgent
from config import AppConfig
from core.orchestrator import DownwellAI
from core.reward_calculator import RewardCalculator
from core.vision import AIVision
from environment.game_env import CustomDownwellEnvironment
from environment.mem_extractor import Player


def get_game_module(proc, executable_name):
    try:
        return module_from_name(proc.process_handle, executable_name).lpBaseOfDll
    except Exception as e:
        print(f"Error finding module '{executable_name}': {str(e)}")
        raise


def main():
    config = AppConfig()

    print("Starting Downwell.AI v1.0")
    print("=" * 50)

    os.makedirs("models", exist_ok=True)

    # Windows check
    if platform.system() != "Windows":
        print("ERROR: This script currently only supports Windows.")
        return

    # Connect to game
    executable_name = "downwell.exe"
    print(f"Connecting to {executable_name}...")

    try:
        proc = pymem.Pymem(executable_name)
        gameModule = get_game_module(proc, executable_name)
        print("Successfully connected to game")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Downwell is running.")
        return

    # Initialize components
    try:
        player = Player(proc, gameModule)
        env = CustomDownwellEnvironment(config=config.env)

        agent = DQNAgent(
            action_space=env.actions,
            config=config.agent,
            env_config=config.env
        )
        agent.memory = deque(maxlen=config.training.memory_size)
        reward_calc = RewardCalculator(config=config.rewards)
        ai_system = DownwellAI(player, env, agent, reward_calc, config=config.env)

    except Exception as e:
        print(f"Error initializing: {e}")
        return

    # Training parameters
    max_episodes = config.training.max_episodes
    save_frequency = config.training.save_frequency
    target_update_frequency = config.training.target_update_frequency
    episode = 0
    best_reward = float('-inf')

    # AI Vision
    vision = AIVision()

    # Plot training history
    training_history = []

    try:
        while episode < max_episodes:
            episode += 1
            print(f"\n--- Episode {episode} ---")

            # Reset environment
            state = env.reset(player)
            if state is None:
                print("Failed to reset, skipping episode")
                continue

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
                        print("Episode ended - Game Over!")
                        time.sleep(0.3)
                        break

            # Stop AI system and get learning statistics
            ai_system.stop()
            episode_stats = ai_system.get_episode_stats()
            episode_duration = time.time() - episode_start

            # Episode summary
            print(f"\nEpisode {episode} Summary:")
            print(f"  Reward: {episode_stats['episode_reward']:.1f}")
            print(f"  Duration: {episode_duration:.1f}s")
            print(f"  Steps: {episode_stats['steps']}")
            print(f"  Max Combo: {max_combo:.0f}")
            print(f"  Final Gems: {final_gems:.0f}")
            print(
                f"  Experiences: +{episode_stats['experiences_added']} (total: {len(agent.memory)}/{agent.memory.maxlen})")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Learning Rate: {agent.scheduler.get_last_lr()[0]:.6f}")

            # Save episode data
            episode_data = {
                'episode': episode,
                'reward': episode_stats['episode_reward'],
                'duration': episode_duration,
                'steps': episode_stats['steps'],
                'max_combo': max_combo,
                'final_gems': final_gems,
                'epsilon': agent.epsilon,
                'learning_rate': agent.scheduler.get_last_lr()[0]
            }
            training_history.append(episode_data)

            episode_reward = episode_stats['episode_reward']
            if episode_reward > best_reward:
                best_reward = episode_reward
                print(f"  🎉 NEW BEST: {best_reward:.2f}")
                agent.save_model("models/downwell_ai_best.pth")

            if episode % save_frequency == 0:
                model_path = f"models/downwell_ai_{episode}.pth"
                agent.save_model(model_path)
                print(f"  💾 Model saved: {model_path}")

            if episode % target_update_frequency == 0:
                agent.update_target_network()
                print("  🎯 Target network updated")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\nCleaning up...")
        if ai_system:
            ai_system.stop()
        vision.close()

        # Save training history
        if training_history:
            print("\nSaving training history...")
            keys = training_history[0].keys()
            with open('training_history_1.csv', 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(training_history)
            print("Saved training history to training_history.csv")

        # Save final model
        try:
            final_model_path = f"models/downwell_ai_final_{episode}.pth"
            agent.save_model(final_model_path)
            print(f"Final model saved: {final_model_path}")
        except Exception as e:
            print(f"Error saving final model: {e}")

        print("Training session ended")
        print(f"Total episodes completed: {episode}")
        print(f"Best episode reward: {best_reward:.2f}")


if __name__ == "__main__":
    main()
