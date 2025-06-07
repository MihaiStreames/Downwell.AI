import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, input_channels=3, num_actions=6):
        super(DQN, self).__init__()

        # Convolutional layers for image processing
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Linear layers
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.action_head = nn.Linear(512, num_actions)  # Which action to take
        self.duration_head = nn.Linear(512, 1)  # How long to hold it

        self.relu = nn.ReLU()

    def forward(self, x):
        # Normalize input to [0, 1]
        x = x.float() / 255.0

        # Conv layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # Adaptive pooling to handle variable sizes
        x = self.adaptive_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.relu(self.fc1(x))

        # Output action probabilities and duration
        actions = self.action_head(x)
        duration = torch.relu(self.duration_head(x)) + 0.02  # Minimum 0.02s, no upper bound
        return actions, duration


class Agent:
    def __init__(self, action_space, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995):
        self.action_space = action_space
        self.action_size = len(action_space)
        self.memory = deque(maxlen=10000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(num_actions=self.action_size).to(self.device)
        self.target_network = DQN(num_actions=self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Update target network
        self.update_target_network()

        # Training parameters
        self.batch_size = 32
        self.train_start = 1000

        print(f"Agent initialized on device: {self.device}")

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, duration, reward, next_state, done):
        self.memory.append((state, action, duration, reward, next_state, done))

    def get_action(self, state):
        if state is None:
            return random.randrange(self.action_size), 0.1

        if random.uniform(0, 1) <= self.epsilon:
            action = random.randrange(self.action_size)
            duration = max(0.02, random.uniform(0, 2.0))  # Random duration between 0.02 and 2 seconds
            return action, duration

        # Convert state to tensor
        state_tensor = self.preprocess_state(state)
        if state_tensor is None:
            return random.randrange(self.action_size), 0.1

        state_tensor = state_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            actions, duration = self.q_network(state_tensor)
            action = torch.argmax(actions).item()
            duration_val = max(0.02, duration.item())  # Ensure minimum duration
            return action, duration_val

    @staticmethod
    def preprocess_state(state):
        try:
            if state is None:
                return None

            # Handle different input shapes
            if len(state.shape) == 3:  # (H, W, C)
                state_tensor = torch.from_numpy(state).permute(2, 0, 1).float()
            elif len(state.shape) == 4:  # (B, H, W, C) - batch dimension
                state_tensor = torch.from_numpy(state).permute(0, 3, 1, 2).float()
            else:
                print(f"Unexpected state shape: {state.shape}")
                return None

            return state_tensor
        except Exception as e:
            print(f"Error preprocessing state: {e}")
            return None

    def train(self, state, action, duration, reward, next_state, done):
        # Store experience in memory
        self.remember(state, action, duration, reward, next_state, done)

        # Train if we have enough experiences
        if len(self.memory) > self.train_start:
            self.replay()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)

        states = []
        actions = []
        durations = []
        rewards = []
        next_states = []
        dones = []

        for experience in batch:
            state, action, duration, reward, next_state, done = experience

            state_tensor = self.preprocess_state(state)
            next_state_tensor = self.preprocess_state(next_state)

            if state_tensor is not None and next_state_tensor is not None:
                states.append(state_tensor)
                actions.append(action)
                durations.append(duration)
                rewards.append(reward)
                next_states.append(next_state_tensor)
                dones.append(done)

        if len(states) == 0:
            return

        # Convert to tensors
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        durations = torch.tensor(durations, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)

        # Current Q values
        current_actions, current_durations = self.q_network(states)
        current_q_values = current_actions.gather(1, actions.unsqueeze(1))

        # Duration loss (MSE between predicted and actual duration)
        duration_loss = nn.MSELoss()(current_durations.squeeze(), durations)

        # Next Q values from target network
        with torch.no_grad():
            next_actions, next_durations = self.target_network(next_states)
            next_q_values = next_actions.max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Action Q-value loss
        action_loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Combined loss
        total_loss = action_loss + 0.1 * duration_loss  # Weight duration loss lower

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filepath):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)

    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
