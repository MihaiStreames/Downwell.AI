import platform
import time
from datetime import datetime

from pymem.process import *

from src.custom_env import CustomDownwellEnvironment
from src.dqn_agent import Agent
from src.mem_extractor import Player


def get_game_module(proc, executable_name):
    try:
        return module_from_name(proc.process_handle, executable_name).lpBaseOfDll
    except Exception as e:
        print(f"Error finding module '{executable_name}': {str(e)}")
        raise


def setup_directories():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)


def save_training_log(episode, reward, steps, epsilon, filename="logs/training_log.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, "a") as f:
        f.write(f"{timestamp} - Episode: {episode}, Reward: {reward:.2f}, Steps: {steps}, Epsilon: {epsilon:.3f}\n")


def main():
    print("Starting Downwell RL Bot v1.0")
    print("=" * 50)

    setup_directories()

    os_type = platform.system()
    if os_type != "Windows":
        print("ERROR: This script currently only supports Windows.")
        print("The game memory reading functionality requires Windows-specific libraries.")
        return

    executable_name = "downwell.exe"

    print(f"Attempting to connect to {executable_name}...")
    try:
        proc = pymem.Pymem(executable_name)
        print("Successfully connected to game process")
    except Exception as e:
        print(f"Error opening process '{executable_name}': {str(e)}")
        print("Make sure Downwell is running and try again.")
        return

    try:
        gameModule = get_game_module(proc, executable_name)
        print("Successfully found game module")
    except Exception as e:
        print(f"Error getting game module: {str(e)}")
        return

    print("Initializing components...")
    try:
        player = Player(proc, gameModule)
        gameEnv = CustomDownwellEnvironment()
        agent = Agent(gameEnv.actions, learning_rate=0.0001, epsilon=1.0, epsilon_decay=0.995)
        print("All components initialized successfully")
    except Exception as e:
        print(f"Error initializing components: {str(e)}")
        return

    print("\nStarting training...")
    print("=" * 50)

    # Training parameters
    max_episodes = 1000
    max_steps_per_episode = 5000
    save_frequency = 10  # Save model every 10 episodes
    target_update_frequency = 100  # Update target network every 100 episodes

    episode = 0

    try:
        while episode < max_episodes:
            episode += 1
            print(f"\n--- Episode {episode} ---")

            if not player.validate_connection():
                print("Lost connection to game. Attempting to reconnect...")
                time.sleep(5)
                continue

            state = gameEnv.reset(player)
            if state is None:
                print("Failed to reset environment, skipping episode")
                continue

            episode_reward = 0
            steps = 0
            start_time = time.time()

            while steps < max_steps_per_episode:
                steps += 1

                # 1. Agent chooses action
                action = agent.get_action(state)

                # 2. Environment executes action
                next_state, reward, done = gameEnv.step(action, player)

                if next_state is None:
                    print("Failed to get next state, ending episode")
                    break

                # 3. Agent learns from experience
                agent.train(state, action, reward, next_state, done)

                # 4. Update state and reward
                state = next_state
                episode_reward += reward

                # Get debug info
                debug_info = gameEnv.get_debug_info(player)

                if steps % 50 == 0:
                    print(f"Step {steps}: Action={gameEnv.actions[action]}, Reward={reward:.1f}, "
                          f"Total={episode_reward:.1f}, HP={debug_info.get('hp', 'N/A')}, "
                          f"Gems={debug_info.get('gems', 'N/A')}, Combo={debug_info.get('combo', 'N/A')}")

                if done:
                    break

            # Episode finished
            episode_time = time.time() - start_time
            print(f"\nEpisode {episode} finished:")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Steps: {steps}")
            print(f"  Time: {episode_time:.1f}s")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Memory size: {len(agent.memory)}")

            save_training_log(episode, episode_reward, steps, agent.epsilon)

            if episode % save_frequency == 0:
                model_path = f"models/dqn_model_episode_{episode}.pth"
                agent.save_model(model_path)
                print(f"Model saved to {model_path}")

            if episode % target_update_frequency == 0:
                agent.update_target_network()
                print("Target network updated")

            time.sleep(2)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error during training: {e}")
    finally:
        # Save final model
        final_model_path = f"models/dqn_model_final_episode_{episode}.pth"
        try:
            agent.save_model(final_model_path)
            print(f"Final model saved to {final_model_path}")
        except Exception as e:
            print(f"Error saving final model: {e}")

        print("\nTraining session ended")
        print(f"Total episodes completed: {episode}")

    if __name__ == "__main__":
        main()
