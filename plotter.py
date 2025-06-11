import csv
import matplotlib.pyplot as plt


def plot_training_history(filepath='training_history.csv'):
    # Initialize lists to hold the data
    episodes, rewards, durations, combos, gems = [], [], [], [], []

    try:
        with open(filepath, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                episodes.append(int(row['episode']))
                rewards.append(float(row['reward']))
                durations.append(float(row['duration']))
                combos.append(int(row['max_combo']))
                gems.append(int(row['final_gems']))
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        print("Please run a training session first to generate the history file.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # Create a figure with 4 subplots that share the same x-axis
    fig, axs = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
    fig.suptitle('Downwell.AI Training Progress', fontsize=16)

    # 1. Plot Total Reward
    axs[0].plot(episodes, rewards, linestyle='-', color='b')
    axs[0].set_ylabel('Total Reward')
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # Also plot a moving average for reward to see the trend
    moving_avg_reward = [sum(rewards[max(0, i - 10):i + 1]) / len(rewards[max(0, i - 10):i + 1]) for i in
                         range(len(rewards))]
    axs[0].plot(episodes, moving_avg_reward, color='r', linestyle='--', label='10-ep Moving Avg')
    axs[0].legend()

    # 2. Plot Episode Duration
    axs[1].plot(episodes, durations, linestyle='-', color='g')
    axs[1].set_ylabel('Duration (s)')
    axs[1].grid(True, linestyle='--', alpha=0.6)

    # 3. Plot Max Combo
    axs[2].plot(episodes, combos, linestyle='-', color='m')
    axs[2].set_ylabel('Max Combo')
    axs[2].grid(True, linestyle='--', alpha=0.6)

    # 4. Plot Final Gems
    axs[3].plot(episodes, gems, linestyle='-', color='orange')
    axs[3].set_ylabel('Final Gems')
    axs[3].grid(True, linestyle='--', alpha=0.6)

    # Set the common X-axis label
    plt.xlabel('Episode')

    plt.savefig('training_progress.png')
    print("Saved plot to training_progress.png")
    plt.show()


if __name__ == '__main__':
    plot_training_history()