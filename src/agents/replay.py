import numpy as np
import torch


class ReplayBuffer:
    """Tensor-based replay buffer"""

    def __init__(self, capacity, state_shape, device=torch.device("cpu")):
        self.capacity = capacity
        self.position = 0
        self.size = 0
        self.device = device

        # Pre-alloc with numpy
        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.memory_features = np.zeros((capacity, 6), dtype=np.float32)
        self.next_memory_features = np.zeros((capacity, 6), dtype=np.float32)

    def add(self, state, action, reward, next_state, done, memory_features, next_memory_features):
        """Add experience"""
        idx = self.position

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        self.memory_features[idx] = memory_features
        self.next_memory_features[idx] = next_memory_features

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample batch"""
        if self.size < batch_size:
            return None

        # Random sampling using numpy
        indices = np.random.randint(0, self.size, size=batch_size)

        # Convert to tensors and move to device in one operation
        batch = (
            torch.from_numpy(self.states[indices]).to(self.device, non_blocking=True),
            torch.from_numpy(self.actions[indices]).to(self.device, non_blocking=True),
            torch.from_numpy(self.rewards[indices]).to(self.device, non_blocking=True),
            torch.from_numpy(self.next_states[indices]).to(self.device, non_blocking=True),
            torch.from_numpy(self.dones[indices]).to(self.device, non_blocking=True),
            torch.from_numpy(self.memory_features[indices]).to(self.device, non_blocking=True),
            torch.from_numpy(self.next_memory_features[indices]).to(self.device, non_blocking=True)
        )

        return batch

    def __len__(self):
        return self.size
