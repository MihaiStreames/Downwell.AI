import argparse
import os
import platform
import time
from collections import deque

import cv2
import pymem
from pymem.process import module_from_name

from agents.dqn_agent import DQNAgent
from core.orchestrator import DownwellAI
from core.reward_calculator import RewardCalculator
from environment.game_env import CustomDownwellEnvironment
from environment.mem_extractor import Player


def get_game_module(proc, executable_name):
    try:
        return module_from_name(proc.process_handle, executable_name).lpBaseOfDll
    except Exception as e:
        print(f"Error finding module '{executable_name}': {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Downwell AI Training')
    parser.add_argument('--load-model', type=str, help='Path to pre-trained model to load')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')
    parser.add_argument('--memory-size', type=int, default=12500, help='Experience replay buffer size')
    parser.add_argument('--learning-rate', type=float, default=0.0005, help='Learning rate')
    args = parser.parse_args()

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
        env = CustomDownwellEnvironment()

        agent = DQNAgent(
            env.actions,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            epsilon=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.9995,
            pretrained_model=args.load_model
        )
        agent.memory = deque(maxlen=args.memory_size)
        reward_calc = RewardCalculator()
        ai_system = DownwellAI(player, env, agent, reward_calc)

    except Exception as e:
        print(f"Error initializing: {e}")
        return

    # Training parameters
    max_episodes = args.episodes
    save_frequency = 25
    target_update_frequency = 100
    episode = 0
    best_reward = float('-inf')

    try:
        while episode < max_episodes:
            episode += 1
            print(f"\n--- Episode {episode} ---")

            # Reset environment
            state = env.reset(player)
            if state is None:
                print("Failed to reset, skipping episode")
                continue

            reward_calc.reset_episode()
            ai_system.start()

            # Monitor episode
            episode_start = time.time()
            steps = 0
            max_combo = 0
            final_gems = 0

            while True:
                current_state = ai_system.get_latest_state()

                if current_state:
                    env.show_ai_vision(
                        current_state.screenshot,
                        f"E{episode} HP:{current_state.hp:.0f} G:{current_state.gems:.0f} C:{current_state.combo:.0f} A:{current_state.ammo:.0f} GH:{current_state.gem_high}",
                        player
                    )

                    max_combo = max(max_combo, current_state.combo)
                    final_gems = current_state.gems

                    if current_state.hp <= 0:
                        print("Episode ended - Game Over!")
                        break

                    steps += 1

                # Timeout check
                if time.time() - episode_start > 300:  # 5 minute timeout
                    print("Episode timeout")
                    break

            ai_system.stop()

            # Train on collected data
            episode_reward, experiences_added = ai_system.train_on_episode()
            episode_duration = time.time() - episode_start

            # Episode summary
            print(f"\nEpisode {episode} Summary:")
            print(f"  Reward: {episode_reward:.1f}")
            print(f"  Duration: {episode_duration:.1f}s")
            print(f"  Steps: {steps}")
            print(f"  Max Combo: {max_combo:.0f}")
            print(f"  Final Gems: {final_gems:.0f}")
            print(f"  Experiences: +{experiences_added} (total: {len(agent.memory)}/{agent.memory.maxlen})")
            print(f"  Epsilon: {agent.epsilon:.4f}")

            if episode_reward > best_reward:
                best_reward = episode_reward
                print(f"  🎉 NEW BEST: {best_reward:.2f}")
                agent.save_model("models/downwell_ai_best.pth")

            if episode % save_frequency == 0:
                model_path = f"models/downwell_ai_{episode}.pth"
                agent.save_model(model_path)

            if episode % target_update_frequency == 0:
                agent.update_target_network()
                print("Target network updated")

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
        cv2.destroyAllWindows()

        # Save final model
        try:
            final_model_path = f"models/downwell_ai_final_{episode}.pth"
            agent.save_model(final_model_path)
            print(f"Final model saved: {final_model_path}")
        except Exception as e:
            print(f"Error saving final model: {e}")

        print("Training session ended")
        print(f"Total episodes completed: {episode}")


if __name__ == "__main__":
    main()
