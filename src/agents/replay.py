import numpy as np
import torch


# action flip map: left(2)<->right(3), left+jump(4)<->right+jump(5), rest unchanged
_FLIP_ACTION_MAP = torch.tensor([0, 1, 3, 2, 5, 4], dtype=torch.int64)


class ReplayBuffer:
    """Pre-allocated circular buffer. Converts to tensors on sample."""

    def __init__(
        self, capacity: int, state_shape: tuple[int, ...], device: torch.device | None = None
    ) -> None:
        self._capacity: int = capacity

        self._position: int = 0
        self._size: int = 0

        self._device: torch.device = device if device is not None else torch.device("cpu")

        self._states: np.ndarray = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self._next_states: np.ndarray = np.zeros((capacity, *state_shape), dtype=np.uint8)

        self._actions: np.ndarray = np.zeros(capacity, dtype=np.int64)

        self._rewards: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self._dones: np.ndarray = np.zeros(capacity, dtype=np.bool_)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return self._size

    def add(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ) -> None:
        idx = self._position
        self._states[idx] = state
        self._actions[idx] = action
        self._rewards[idx] = reward
        self._next_states[idx] = next_state
        self._dones[idx] = done
        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if self._size < batch_size:
            return None

        indices = np.random.randint(0, self._size, size=batch_size)
        states = torch.from_numpy(self._states[indices]).to(self._device, non_blocking=True)
        actions = torch.from_numpy(self._actions[indices]).to(self._device, non_blocking=True)
        rewards = torch.from_numpy(self._rewards[indices]).to(self._device, non_blocking=True)
        next_states = torch.from_numpy(self._next_states[indices]).to(
            self._device, non_blocking=True
        )
        dones = torch.from_numpy(self._dones[indices]).to(self._device, non_blocking=True)

        # ~50% of samples mirrored left/right
        flip_mask = torch.rand(batch_size, device=self._device) < 0.5
        if flip_mask.any():
            # states are (B, H, W, C); axis 2 is width
            states = torch.where(flip_mask[:, None, None, None], states.flip(2), states)
            next_states = torch.where(
                flip_mask[:, None, None, None], next_states.flip(2), next_states
            )
            flip_map = _FLIP_ACTION_MAP.to(self._device)
            actions = torch.where(flip_mask, flip_map[actions], actions)

        return (states, actions, rewards, next_states, dones)
