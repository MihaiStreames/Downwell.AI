import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler

from agents.dqn_network import DQN
from agents.replay import ReplayBuffer
from config import AgentConfig, EnvConfig


class DQNAgent:
    def __init__(self, action_space: dict, config: AgentConfig, env_config: EnvConfig):
        self.action_space = action_space
        self.action_size = len(action_space)

        # Config
        self.gamma = config.gamma
        self.epsilon = config.epsilon_start
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.train_start = config.train_start

        # RTX GPU
        self.device = torch.device("cuda")
        self.scaler = GradScaler("cuda")

        # Replay buffer
        self.memory = ReplayBuffer(
            capacity=500000,
            state_shape=(env_config.image_size[0], env_config.image_size[1], env_config.frame_stack),
            device=self.device
        )

        # Networks
        self.q_network = DQN(
            input_channels=env_config.frame_stack,
            num_actions=self.action_size
        ).to(self.device)

        self.target_network = DQN(
            input_channels=env_config.frame_stack,
            num_actions=self.action_size
        ).to(self.device)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate, eps=1e-8)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.9)

        if config.pretrained_model and os.path.exists(config.pretrained_model):
            self.load_model(config.pretrained_model)
            print(f"Loaded pre-trained model from: {config.pretrained_model}")
        else:
            self.update_target_network()

        print(f"Agent initialized on: {torch.cuda.get_device_name()}")
        print(f"  Gamma: {self.gamma}")
        print(f"  Memory: {self.memory.capacity:,}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Training starts: {self.train_start:,}")
        print(f"  Epsilon: {self.epsilon:.3f} → {self.epsilon_min:.3f}")

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def get_action(self, game_state):
        if game_state is None:
            return random.randrange(self.action_size), np.zeros(self.action_size)

        if random.uniform(0, 1) <= self.epsilon:
            return random.randrange(self.action_size), np.zeros(self.action_size)

        state = game_state.screenshot
        if state is None:
            return random.randrange(self.action_size), np.zeros(self.action_size)

        # Direct tensor conversion
        state_tensor = torch.from_numpy(state).permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)

        with torch.no_grad():
            actions = self.q_network(state_tensor)

        return torch.argmax(actions).item(), actions.cpu().numpy().flatten()

    def train(self, state, action, reward, next_state, done):
        self.remember(state.screenshot, action, reward, next_state.screenshot, done)

        if len(self.memory) >= self.train_start:
            return self.replay()
        return None

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return None

        state_tensors, action_tensors, reward_tensors, next_state_tensors, done_tensors = batch

        # Preprocess: permute from (B, H, W, C) to (B, C, H, W)
        state_tensors = state_tensors.permute(0, 3, 1, 2)
        next_state_tensors = next_state_tensors.permute(0, 3, 1, 2)

        # Mixed precision forward pass
        with autocast("cuda"):
            # Current Q-values
            current_q_values = self.q_network(state_tensors).gather(
                1, action_tensors.unsqueeze(-1)
            )

            # Target Q-values
            with torch.no_grad():
                next_q_values = self.target_network(next_state_tensors).max(1)[0]
                target_q_values = reward_tensors + (self.gamma * next_q_values * ~done_tensors)

            # Loss
            loss = nn.SmoothL1Loss()(current_q_values.squeeze(), target_q_values)

        # Backward pass with gradient scaling
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)

        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        # Decay epsilon
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
            'memory_size': len(self.memory)
        }, filepath)

    def load_model(self, filepath):
        try:
            checkpoint = torch.load(filepath, map_location=self.device)

            self.q_network.load_state_dict(checkpoint.get('q_network_state_dict', checkpoint), strict=False)
            self.target_network.load_state_dict(checkpoint.get('target_network_state_dict', checkpoint), strict=False)

            if 'optimizer_state_dict' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except Exception as e:
                    print(f"Optimizer state incompatible: {e}")

            self.epsilon = max(checkpoint.get('epsilon', 0.2), self.epsilon_min)
            self.gamma = checkpoint.get('gamma', self.gamma)

            print(f"Model loaded. Epsilon: {self.epsilon:.3f}")

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting with fresh model...")
