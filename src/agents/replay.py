import numpy as np
import torch


class ReplayBuffer:
    """Pre-allocated circular buffer. Converts to tensors on sample."""

    def __init__(self, capacity, state_shape, device=None):
        self.capacity = capacity
        self.position = 0
        self.size = 0
        self.device = device if device is not None else torch.device("cpu")

        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    def add(self, state, action, reward, next_state, done):
        idx = self.position
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size < batch_size:
            return None

        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[indices]).to(self.device, non_blocking=True),
            torch.from_numpy(self.actions[indices]).to(self.device, non_blocking=True),
            torch.from_numpy(self.rewards[indices]).to(self.device, non_blocking=True),
            torch.from_numpy(self.next_states[indices]).to(self.device, non_blocking=True),
            torch.from_numpy(self.dones[indices]).to(self.device, non_blocking=True),
        )

    def __len__(self):
        return self.size
