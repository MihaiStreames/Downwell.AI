import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.dqn_network import DQN
from config import AgentConfig, EnvConfig


class DQNAgent:
    def __init__(self, action_space: dict, config: AgentConfig, env_config: EnvConfig):
        self.action_space = action_space
        self.action_size = len(action_space)

        # Set attributes from the config
        self.gamma = config.gamma
        self.epsilon = config.epsilon_start
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.train_start = config.train_start

        self.memory = deque(maxlen=100)  # This will be overwritten in main.py

        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(input_channels=env_config.frame_stack, num_actions=self.action_size, memory_features=6).to(self.device)
        self.target_network = DQN(input_channels=env_config.frame_stack, num_actions=self.action_size, memory_features=6).to(self.device)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate, eps=1e-8)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.9)

        if config.pretrained_model and os.path.exists(config.pretrained_model):
            self.load_model(config.pretrained_model)
            print(f"Loaded pre-trained model from: {config.pretrained_model}")
        else:
            self.update_target_network()

        # Training parameters
        self.batch_size = 64
        self.train_start = 10000

        print(f"Agent initialized on device: {self.device}")
        print(f"  Gamma: {self.gamma}")
        print(f"  Memory: {self.memory.maxlen:,}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Training starts: {self.train_start}")
        print(f"  Epsilon range: {self.epsilon:.3f} → {self.epsilon_min}")

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done, memory_features, next_memory_features):
        self.memory.append((state, action, reward, next_state, done, memory_features, next_memory_features))

    @staticmethod
    def extract_memory_features(game_state):
        if game_state is None: return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Use default values for None to prevent errors during feature extraction
        hp = game_state.hp if game_state.hp is not None and game_state.hp != 999.0 else 8.0
        xpos = game_state.xpos if game_state.xpos is not None else 0.0
        ypos = game_state.ypos if game_state.ypos is not None else 0.0

        features = np.array([
            hp,
            game_state.gems,
            game_state.combo,
            xpos,
            ypos,
            getattr(game_state, 'ammo', 0.0)
        ], dtype=np.float32)

        return features

    def get_action(self, game_state):
        if game_state is None:
            return random.randrange(self.action_size), np.zeros(self.action_size)

        memory_features = self.extract_memory_features(game_state)

        if random.uniform(0, 1) <= self.epsilon:
            return random.randrange(self.action_size), np.zeros(self.action_size)

        state_tensor = self.preprocess_state(game_state.screenshot)
        if state_tensor is None:
            return random.randrange(self.action_size), np.zeros(self.action_size)

        state_tensor = state_tensor.unsqueeze(0).to(self.device)
        memory_tensor = torch.from_numpy(memory_features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            actions = self.q_network(state_tensor, memory_tensor)

        return torch.argmax(actions).item(), actions.cpu().numpy().flatten()

    @staticmethod
    def preprocess_state(state):
        if state is None: return None
        # The environment is responsible for shape and channel count
        # The agent just needs to convert it to a tensor
        # Permute (H, W, C) to (C, H, W) for PyTorch
        return torch.from_numpy(state).permute(2, 0, 1).float()

    def train(self, state, action, reward, next_state, done, memory_features, next_memory_features):
        self.remember(state.screenshot, action, reward, next_state.screenshot, done, memory_features, next_memory_features)

        # Train if we have enough experiences
        if len(self.memory) > self.train_start: return self.replay()
        return None

    def replay(self):
        if len(self.memory) < self.batch_size: return None

        batch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, dones, memory_features, next_memory_features = zip(*batch)

        state_tensors = torch.stack([self.preprocess_state(s) for s in states]).to(self.device)
        next_state_tensors = torch.stack([self.preprocess_state(s) for s in next_states]).to(self.device)
        action_tensors = torch.tensor(actions, dtype=torch.long).to(self.device)
        reward_tensors = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        done_tensors = torch.tensor(dones, dtype=torch.bool).to(self.device)
        memory_features_tensors = torch.from_numpy(np.array(memory_features)).float().to(self.device)
        next_memory_features_tensors = torch.from_numpy(np.array(next_memory_features)).float().to(self.device)

        current_q_values = self.q_network(state_tensors, memory_features_tensors).gather(1, action_tensors.unsqueeze(-1))

        with torch.no_grad():
            next_q_values = self.target_network(next_state_tensors, next_memory_features_tensors).max(1)[0]
            target_q_values = reward_tensors + (self.gamma * next_q_values * ~done_tensors)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def save_model(self, filepath):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'memory_size': self.memory.maxlen
        }, filepath)

    def load_model(self, filepath):
        try:
            checkpoint = torch.load(filepath, map_location=self.device)

            self.q_network.load_state_dict(checkpoint.get('q_network_state_dict', checkpoint), strict=False)
            self.target_network.load_state_dict(checkpoint.get('target_network_state_dict', checkpoint), strict=False)

            if 'optimizer_state_dict' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except:
                    print("Optimizer state incompatible, using fresh optimizer")

            self.epsilon = max(checkpoint.get('epsilon', 0.2), 0.1)
            self.gamma = checkpoint.get('gamma', self.gamma)

            print(f"Model loaded successfully. Epsilon: {self.epsilon:.3f}")

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Continuing with randomly initialized model...")
