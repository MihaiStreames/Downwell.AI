import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.dqn_network import DQN


class DQNAgent:
    def __init__(self, action_space, learning_rate=0.0005, gamma=0.92, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.9995, pretrained_model=None):
        self.action_space = action_space
        self.action_size = len(action_space)

        self.memory = deque(maxlen=12500)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(input_channels=6, num_actions=self.action_size, memory_features=6).to(self.device)
        self.target_network = DQN(input_channels=6, num_actions=self.action_size, memory_features=6).to(self.device)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate, eps=1e-8)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2000, gamma=0.9)

        if pretrained_model and os.path.exists(pretrained_model):
            self.load_model(pretrained_model)
            print(f"Loaded pre-trained model from: {pretrained_model}")
        else:
            # Update target network for fresh models
            self.update_target_network()

        # Training parameters
        self.batch_size = 64
        self.train_start = 500

        print(f"Agent initialized on device: {self.device}")
        print(f"  Gamma: {self.gamma}")
        print(f"  Memory: {self.memory.maxlen:,}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Training starts: {self.train_start}")
        print(f"  Epsilon range: {self.epsilon:.3f} → {self.epsilon_min}")

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, duration, reward, next_state, done,
                 memory_features, next_memory_features):
        self.memory.append((state, action, duration, reward, next_state, done,
                            memory_features, next_memory_features))

    @staticmethod
    def extract_memory_features(game_state):
        if game_state is None:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        features = np.array([
            game_state.hp if game_state.hp != 999.0 else 8.0,  # Use high HP during transitions
            game_state.gems,
            game_state.combo,
            game_state.xpos,
            game_state.ypos,
            getattr(game_state, 'ammo', 0.0)
        ], dtype=np.float32)

        return features

    def get_action(self, game_state, episode_step_count=0):
        if game_state is None:
            action = random.randrange(self.action_size)
            duration = random.uniform(0.02, 0.2)
            return action, duration, np.zeros(self.action_size)

        # Extract memory features
        memory_features = self.extract_memory_features(game_state)

        # Show training progress
        memory_status = f"MEM: {len(self.memory)}/{self.memory.maxlen}"
        training_status = "TRAINING" if len(self.memory) >= self.train_start else f"COLLECTING"

        if random.uniform(0, 1) <= self.epsilon:
            action = random.randrange(self.action_size)
            duration = random.uniform(0.02, 0.25)
            if episode_step_count % 10 == 0:
                print(f"Random: a={action} d={duration:.2f} ε={self.epsilon:.3f} [{training_status}] [{memory_status}]")
                print(f"  Memory: HP={memory_features[0]:.1f} G={memory_features[1]:.0f} C={memory_features[2]:.0f}")
            return action, duration, np.zeros(self.action_size)

        # Convert state to tensor
        state_tensor = self.preprocess_state(game_state.screenshot)
        if state_tensor is None:
            print("Failed to preprocess state!")
            action = random.randrange(self.action_size)
            duration = 0.1
            return action, duration, np.zeros(self.action_size)

        state_tensor = state_tensor.unsqueeze(0).to(self.device)
        memory_tensor = torch.from_numpy(memory_features).unsqueeze(0).to(self.device)

        try:
            with torch.no_grad():
                actions, duration_pred = self.q_network(state_tensor, memory_tensor)
                action = torch.argmax(actions).item()
                duration_val = max(0.02, min(0.3, duration_pred.item()))

                q_values = actions.cpu().numpy().flatten()
                if episode_step_count % 10 == 0:
                    print(
                        f"Network: a={action} d={duration_val:.2f} maxQ={max(q_values):.1f} [{training_status}] [{memory_status}]")
                    print(
                        f"  Memory: HP={memory_features[0]:.1f} G={memory_features[1]:.0f} C={memory_features[2]:.0f}")

                return action, duration_val, q_values
        except Exception as e:
            print(f"Error in neural network forward pass: {e}")
            action = random.randrange(self.action_size)
            duration = 0.1
            return action, duration, np.zeros(self.action_size)

    @staticmethod
    def preprocess_state(state):
        try:
            if state is None:
                return None

            if len(state.shape) == 3:
                if state.shape[2] == 6:
                    state_tensor = torch.from_numpy(state).permute(2, 0, 1).float()
                elif state.shape[2] == 3:
                    rgb = torch.from_numpy(state).permute(2, 0, 1).float()
                    state_tensor = torch.cat([rgb, rgb], dim=0)
                else:
                    print(f"Unexpected number of channels: {state.shape[2]}")
                    return None
            else:
                print(f"Unexpected state shape: {state.shape}")
                return None

            return state_tensor
        except Exception as e:
            print(f"Error preprocessing state: {e}")
            return None

    def train(self, state, action, duration, reward, next_state, done,
              memory_features, next_memory_features):
        # Store experience in memory
        self.remember(state.screenshot, action, duration, reward, next_state.screenshot,
                      done, memory_features, next_memory_features)

        # Train if we have enough experiences
        if len(self.memory) > self.train_start:
            loss = self.replay()
            return loss
        return None

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None

        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)

        states = []
        actions = []
        durations = []
        rewards = []
        next_states = []
        dones = []
        memory_features_batch = []
        next_memory_features_batch = []

        for experience in batch:
            (state, action, duration, reward, next_state, done,
             memory_features, next_memory_features) = experience

            state_tensor = self.preprocess_state(state)
            next_state_tensor = self.preprocess_state(next_state)

            if state_tensor is not None and next_state_tensor is not None:
                states.append(state_tensor)
                actions.append(action)
                durations.append(duration)
                rewards.append(reward)
                next_states.append(next_state_tensor)
                dones.append(done)
                memory_features_batch.append(memory_features)
                next_memory_features_batch.append(next_memory_features)

        if len(states) == 0:
            return None

        # Convert to tensors
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        durations = torch.tensor(durations, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        memory_features_batch = torch.from_numpy(np.array(memory_features_batch)).float().to(self.device)
        next_memory_features_batch = torch.from_numpy(np.array(next_memory_features_batch)).float().to(self.device)

        # Current Q values
        current_actions, current_durations = self.q_network(states, memory_features_batch)
        current_q_values = current_actions.gather(1, actions.unsqueeze(1))

        # Duration loss
        duration_loss = nn.MSELoss()(current_durations.squeeze(), durations)

        # Next Q values from target network
        with torch.no_grad():
            next_actions, next_durations = self.target_network(next_states, next_memory_features_batch)
            next_q_values = next_actions.max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Action Q-value loss
        action_loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Combined loss
        total_loss = action_loss + 0.1 * duration_loss

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return total_loss.item()

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
