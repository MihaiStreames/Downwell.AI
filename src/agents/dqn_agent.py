from pathlib import Path
import random

from loguru import logger
import numpy as np
import torch
from torch import nn
from torch import optim

from src.config import Config
from src.models.game_state import GameState
from src.utils.consts import ADAM_EPS
from src.utils.consts import LR_SCHEDULER_GAMMA

from .dqn_network import DQN
from .replay import ReplayBuffer


class DQNAgent:
    def _update_target_network(self) -> None:
        for tp, op in zip(
            self._target_network.parameters(), self._q_network.parameters(), strict=False
        ):
            tp.data.copy_(self._tau * op.data + (1.0 - self._tau) * tp.data)

    def _load_model(self, filepath: str) -> None:
        try:
            checkpoint = torch.load(filepath, map_location=self._device, weights_only=False)

            self._q_network.load_state_dict(
                checkpoint.get("q_network_state_dict", checkpoint), strict=False
            )
            self._target_network.load_state_dict(
                checkpoint.get("target_network_state_dict", checkpoint), strict=False
            )

            if "optimizer_state_dict" in checkpoint:
                try:
                    self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                except Exception as e:
                    logger.warning(f"Optimizer state incompatible: {e}")

            self._epsilon = max(checkpoint.get("epsilon", 0.2), self._epsilon_min)
            self._gamma = checkpoint.get("gamma", self._gamma)

            logger.success(f"Model loaded. Epsilon: {self._epsilon:.3f}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Starting with fresh model...")

    def __init__(self, action_space: dict[int, set[str]], config: Config) -> None:
        self._action_size: int = len(action_space)

        self._gamma: float = config.gamma
        self._learning_rate: float = config.learning_rate
        self._batch_size: int = config.batch_size
        self._train_start: int = config.train_start
        self._tau: float = config.target_update_tau
        self._grad_clip_norm: float = config.grad_clip_norm
        self._epsilon_min: float = config.epsilon_min
        self._epsilon_decay: float = config.epsilon_decay

        self._device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._memory: ReplayBuffer = ReplayBuffer(
            capacity=config.memory_size,
            state_shape=(
                config.image_size[0],
                config.image_size[1],
                config.frame_stack,
            ),
            device=self._device,
            flip_probability=config.replay_flip_probability,
        )

        self._q_network: DQN = DQN(
            input_channels=config.frame_stack, num_actions=self._action_size
        ).to(self._device)

        self._target_network: DQN = DQN(
            input_channels=config.frame_stack, num_actions=self._action_size
        ).to(self._device)

        self._optimizer: optim.Adam = optim.Adam(
            self._q_network.parameters(), lr=self._learning_rate, eps=ADAM_EPS
        )

        self.scheduler: optim.lr_scheduler.StepLR = optim.lr_scheduler.StepLR(
            self._optimizer, step_size=config.lr_step_size, gamma=LR_SCHEDULER_GAMMA
        )

        self._epsilon: float = config.epsilon_start

        if config.pretrained_model and Path(config.pretrained_model).exists():
            self._load_model(config.pretrained_model)
            logger.success(f"Loaded pre-trained model from: {config.pretrained_model}")
        else:
            self._update_target_network()

        logger.info(f"Agent initialized on: {torch.cuda.get_device_name()}")
        logger.info(f"Gamma: {self._gamma}")
        logger.info(f"Memory: {self._memory.capacity:,}")
        logger.info(f"Batch size: {self._batch_size}")
        logger.info(f"Training starts: {self._train_start:,}")
        logger.info(f"Epsilon: {self._epsilon:.3f} -> {self._epsilon_min:.3f}")

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def memory(self) -> ReplayBuffer:
        return self._memory

    def _remember(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ) -> None:
        self._memory.add(state, action, reward, next_state, done)

    def get_action(self, game_state: GameState | None) -> tuple[int, np.ndarray]:
        if game_state is None:
            return random.randrange(self._action_size), np.zeros(self._action_size)

        if random.uniform(0, 1) <= self._epsilon:
            return random.randrange(self._action_size), np.zeros(self._action_size)

        state = game_state.screenshot
        if state is None:
            return random.randrange(self._action_size), np.zeros(self._action_size)

        state_tensor = (
            torch.from_numpy(state)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self._device, non_blocking=True)
        )

        with torch.no_grad():
            actions = self._q_network(state_tensor)

        return int(torch.argmax(actions).item()), actions.cpu().numpy().flatten()

    def _replay(self) -> float | None:
        if self._memory.size < self._batch_size:
            return None

        batch = self._memory.sample(self._batch_size)
        if batch is None:
            return None

        (
            state_tensors,
            action_tensors,
            reward_tensors,
            next_state_tensors,
            done_tensors,
        ) = batch

        # permute from (B, H, W, C) to (B, C, H, W)
        state_tensors = state_tensors.permute(0, 3, 1, 2)
        next_state_tensors = next_state_tensors.permute(0, 3, 1, 2)

        current_q_values = self._q_network(state_tensors).gather(1, action_tensors.unsqueeze(-1))

        # Double DQN: online net selects action, target net evaluates
        with torch.no_grad():
            next_actions = self._q_network(next_state_tensors).argmax(1, keepdim=True)
            next_q_values = (
                self._target_network(next_state_tensors).gather(1, next_actions).squeeze(1)
            )
            target_q_values = reward_tensors + (self._gamma * next_q_values * ~done_tensors)

        loss = nn.SmoothL1Loss()(current_q_values.squeeze(), target_q_values)

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._q_network.parameters(), self._grad_clip_norm)
        self._optimizer.step()
        self.scheduler.step()
        self._update_target_network()

        if self._epsilon > self._epsilon_min:
            self._epsilon *= self._epsilon_decay

        return loss.item()

    def train(
        self, state: GameState, action: int, reward: float, next_state: GameState, done: bool
    ) -> float | None:
        if state.screenshot is None or next_state.screenshot is None:
            return None
        self._remember(state.screenshot, action, reward, next_state.screenshot, done)

        if self._memory.size >= self._train_start:
            return self._replay()
        return None

    def save_model(self, filepath: str) -> None:
        torch.save(
            {
                "q_network_state_dict": self._q_network.state_dict(),
                "target_network_state_dict": self._target_network.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "epsilon": self._epsilon,
                "gamma": self._gamma,
                "memory_size": self._memory.size,
            },
            filepath,
        )
