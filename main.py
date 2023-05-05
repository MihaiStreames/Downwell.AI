from customEnv import CustomDownwellEnvironment
from dqnAgent import DQNAgent

def main():
    game_env = CustomDownwellEnvironment()
    ai_agent = DQNAgent(game_env)

    n_episodes = 5000
    for episode in range(n_episodes):
        state = game_env.reset()
        done = False

        while not done:
            action = ai_agent.choose_action(state)

            next_state, reward, done = game_env.step(action)
            ai_agent.remember(state, action, reward, next_state, done)

            state = next_state
            ai_agent.learn()

        print(f"Episode {episode + 1}/{n_episodes} completed.")

    ai_agent.save("downwell_ai_agent.h5")

if __name__ == "__main__":
    main()