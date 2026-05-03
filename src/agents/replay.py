import numpy as np
import torch


class ReplayBuffer:
    """Pre-allocated circular buffer. Converts to tensors on sample."""

    def __init__(self, capacity, state_shape, device=None):
        self.capacity = capacity

        self._position = 0
        self._size = 0

        self._device = device if device is not None else torch.device("cpu")

        self._states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self._actions = np.zeros(capacity, dtype=np.int32)
        self._rewards = np.zeros(capacity, dtype=np.float32)

        self._next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self._dones = np.zeros(capacity, dtype=np.bool_)

    def add(self, state, action, reward, next_state, done):
        idx = self._position
        self._states[idx] = state
        self._actions[idx] = action
        self._rewards[idx] = reward
        self._next_states[idx] = next_state
        self._dones[idx] = done
        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size):
        if self._size < batch_size:
            return None

        indices = np.random.randint(0, self._size, size=batch_size)
        return (
            torch.from_numpy(self._states[indices]).to(self._device, non_blocking=True),
            torch.from_numpy(self._actions[indices]).to(self._device, non_blocking=True),
            torch.from_numpy(self._rewards[indices]).to(self._device, non_blocking=True),
            torch.from_numpy(self._next_states[indices]).to(self._device, non_blocking=True),
            torch.from_numpy(self._dones[indices]).to(self._device, non_blocking=True),
        )

    def __len__(self):
        return self._size
