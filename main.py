import argparse
import platform
import time

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


def main():
    parser = argparse.ArgumentParser(description='Downwell AI Training')
    parser.add_argument('--load-model', type=str, help='Path to pre-trained model to load')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train')
    args = parser.parse_args()

    print("Starting Downwell.AI v1.0")
    print("=" * 50)

    os.makedirs("models", exist_ok=True)

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
        agent = Agent(gameEnv.actions, pretrained_model=args.load_model)
        print("All components initialized successfully")
    except Exception as e:
        print(f"Error initializing components: {str(e)}")
        return

    if args.load_model:
        print(f"Loaded pre-trained model: {args.load_model}")
        print(f"Starting epsilon: {agent.epsilon:.3f}")

    print("\nStarting training...")
    print("=" * 50)

    # Training parameters
    max_episodes = args.episodes
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
            durations = []

            while steps < max_steps_per_episode:
                steps += 1

                # 1. Agent chooses action
                action, duration = agent.get_action(state)
                durations.append(duration)

                # 2. Environment executes action
                next_state, reward, done = gameEnv.step((action, duration), player)

                if next_state is None:
                    print("Failed to get next state, ending episode")
                    break

                # 3. Agent learns from experience
                agent.train(state, action, duration, reward, next_state, done)

                # 4. Update state and reward
                state = next_state
                episode_reward += reward

                # Get debug info
                debug_info = gameEnv.get_debug_info(player)

                if steps % 50 == 0:
                    avg_duration = sum(durations[-50:]) / min(50, len(durations))
                    print(f"Step {steps}: Action={gameEnv.actions[action]}({duration:.2f}s), "
                          f"Reward={reward:.1f}, Total={episode_reward:.1f}, "
                          f"HP={debug_info.get('hp', 'N/A')}, "
                          f"Gems={debug_info.get('gems', 'N/A')}, "
                          f"Combo={debug_info.get('combo', 'N/A')}, "
                          f"AvgDur={avg_duration:.2f}s")

                if done:
                    break

            # Episode finished
            episode_time = time.time() - start_time
            avg_episode_duration = sum(durations) / len(durations) if durations else 0

            print(f"\nEpisode {episode} finished:")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Steps: {steps}")
            print(f"  Time: {episode_time:.1f}s")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Memory size: {len(agent.memory)}")
            print(f"  Average action duration: {avg_episode_duration:.3f}s")

            if episode % save_frequency == 0:
                model_path = f"models/downwell_ai_{episode}.pth"
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
        final_model_path = f"models/downwell_ai_final_{episode}.pth"
        try:
            agent.save_model(final_model_path)
            print(f"Final model saved to {final_model_path}")
        except Exception as e:
            print(f"Error saving final model: {e}")

        # Clean up OpenCV windows
        import cv2
        cv2.destroyAllWindows()

        print("\nTraining session ended")
        print(f"Total episodes completed: {episode}")


if __name__ == "__main__":
    main()
