import numpy as np
import torch


class ReplayBuffer:
    """Tensor-based replay buffer for experience replay.

    Stores transitions in pre-allocated NumPy arrays for efficient memory usage
    and fast random sampling. Automatically converts to PyTorch tensors on sampling.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    state_shape : tuple
        Shape of state observations (height, width, channels).
    device : torch.device, optional
        PyTorch device for tensor conversion (default: CPU).

    Attributes
    ----------
    capacity : int
        Maximum buffer size.
    position : int
        Current write position in circular buffer.
    size : int
        Current number of stored transitions.
    device : torch.device
        Device for tensor operations.
    states : np.ndarray
        Pre-allocated array for states.
    actions : np.ndarray
        Pre-allocated array for actions.
    rewards : np.ndarray
        Pre-allocated array for rewards.
    next_states : np.ndarray
        Pre-allocated array for next states.
    dones : np.ndarray
        Pre-allocated array for done flags.
    """

    def __init__(self, capacity, state_shape, device=None):
        self.capacity = capacity
        self.position = 0
        self.size = 0
        self.device = device if device is not None else torch.device("cpu")

        # Pre-alloc with numpy
        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    def add(self, state, action, reward, next_state, done):
        """Add a transition to the replay buffer.

        Uses circular buffer strategy - oldest experiences are overwritten
        when capacity is reached.

        Parameters
        ----------
        state : np.ndarray
            Current state observation.
        action : int
            Action taken.
        reward : float
            Reward received.
        next_state : np.ndarray
            Resulting state observation.
        done : bool
            Whether episode ended.
        """
        idx = self.position

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample a random batch of transitions.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.

        Returns
        -------
        tuple[torch.Tensor, ...] | None
            Tuple of (states, actions, rewards, next_states, dones) as tensors
            on the configured device, or None if insufficient samples.
        """
        if self.size < batch_size:
            return

        # Random sampling using numpy
        indices = np.random.randint(0, self.size, size=batch_size)

        # Convert to tensors and move to device in one operation
        (
            torch.from_numpy(self.states[indices]).to(self.device, non_blocking=True),
            torch.from_numpy(self.actions[indices]).to(self.device, non_blocking=True),
            torch.from_numpy(self.rewards[indices]).to(self.device, non_blocking=True),
            torch.from_numpy(self.next_states[indices]).to(self.device, non_blocking=True),
            torch.from_numpy(self.dones[indices]).to(self.device, non_blocking=True),
        )

    def __len__(self):
        """Get current number of stored transitions.

        Returns
        -------
        int
            Number of transitions currently in buffer.
        """
        return self.size
