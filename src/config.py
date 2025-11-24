from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """Weights and parameters for the reward function.

    Attributes
    ----------
    level_complete_bonus : float
        Reward for completing a level (default: 100.0).
    death_penalty : float
        Penalty for dying (default: -50.0).
    depth_reward : float
        Reward multiplier per unit of depth progress (default: 2.0).
    gem_reward : float
        Reward per gem collected (default: 1.0).
    combo_threshold : int
        Minimum combo count to trigger bonus (default: 4).
    combo_bonus_multiplier : float
        Multiplier for combo bonus reward (default: 5.0).
    step_penalty : float
        Small penalty per step to encourage efficiency (default: -0.01).
    damage_penalty : float
        Penalty multiplier per HP lost (default: -2.0).
    min_reward_clip : float
        Minimum reward value after clipping (default: -100.0).
    max_reward_clip : float
        Maximum reward value after clipping (default: 100.0).
    """

    level_complete_bonus: float = 100.0
    death_penalty: float = -50.0

    depth_reward: float = 2.0
    gem_reward: float = 1.0

    combo_threshold: int = 4
    combo_bonus_multiplier: float = 5.0

    step_penalty: float = -0.01
    damage_penalty: float = -2.0

    min_reward_clip: float = -100.0
    max_reward_clip: float = 100.0


@dataclass(frozen=True)
class AgentConfig:
    """Hyperparameters for the DQN Agent.

    Attributes
    ----------
    learning_rate : float
        Learning rate for optimizer (default: 0.0001).
    gamma : float
        Discount factor for future rewards (default: 0.9997).
    epsilon_start : float
        Initial exploration rate (default: 1.0).
    epsilon_min : float
        Minimum exploration rate (default: 0.1).
    epsilon_decay : float
        Multiplicative decay rate for epsilon per step (default: 0.999985).
    train_start : int
        Number of experiences before training begins (default: 5000).
    batch_size : int
        Batch size for training (default: 512).
    pretrained_model : str
        Path to pretrained model weights (default: "models/downwell_ai_best.pth").
    """

    learning_rate: float = 0.0001
    gamma: float = 0.9997
    epsilon_start: float = 1.0
    epsilon_min: float = 0.1
    epsilon_decay: float = 0.999985
    train_start: int = 5000
    batch_size: int = 512
    pretrained_model: str = "models/downwell_ai_best.pth"


@dataclass(frozen=True)
class TrainConfig:
    """Parameters for the main training loop.

    Attributes
    ----------
    max_episodes : int
        Maximum number of training episodes (default: 5000).
    memory_size : int
        Replay buffer capacity (default: 100000).
    target_update_frequency : int
        Episodes between target network updates (default: 50).
    save_frequency : int
        Episodes between model checkpoints (default: 25).
    """

    """Parameters for the main training loop"""

    max_episodes: int = 5000
    memory_size: int = 100000
    target_update_frequency: int = 50
    save_frequency: int = 25


@dataclass(frozen=True)
class EnvConfig:
    """Configuration for the game environment and threads.

    Attributes
    ----------
    image_size : tuple[int, int]
        Target dimensions for processed frames (default: (84, 84)).
    frame_stack : int
        Number of consecutive frames to stack as state (default: 4).
    perceptor_fps : int
        Target frames per second for perception thread (default: 60).
    thinker_fps : int
        Target decisions per second for thinker thread (default: 15).
    """

    image_size: tuple[int, int] = (84, 84)
    frame_stack: int = 4
    perceptor_fps: int = 60
    thinker_fps: int = 15


@dataclass(frozen=True)
class AppConfig:
    """Root configuration object for the Downwell AI system.

    Attributes
    ----------
    agent : AgentConfig
        DQN agent hyperparameters.
    rewards : RewardConfig
        Reward function weights and parameters.
    training : TrainConfig
        Training loop configuration.
    env : EnvConfig
        Environment and threading configuration.
    """

    agent: AgentConfig = AgentConfig()
    rewards: RewardConfig = RewardConfig()
    training: TrainConfig = TrainConfig()
    env: EnvConfig = EnvConfig()
