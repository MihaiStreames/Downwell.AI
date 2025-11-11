from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """Weights and parameters for the reward function"""

    # Core events
    level_complete_bonus: float = 100.0
    death_penalty: float = -10.0

    # Main rewards
    depth_reward: float = 0.5
    survival_reward: float = 0.01

    # Damage penalty
    damage_penalty: float = -0.5

    # Immobility penalties
    immobility_grace_period: int = 60
    immobility_base_penalty: float = 0.01
    immobility_growth_rate: float = 1.05
    immobility_max_penalty: float = 5.0

    # Clipping
    min_reward_clip: float = -20.0
    max_reward_clip: float = 20.0


@dataclass(frozen=True)
class AgentConfig:
    """Hyperparameters for the DQN Agent"""

    learning_rate: float = 0.0001
    gamma: float = 0.85
    epsilon_start: float = 1.0
    epsilon_min: float = 0.1
    epsilon_decay: float = 0.999985
    train_start: int = 5000
    batch_size: int = 128
    pretrained_model: str = "models/downwell_ai_best.pth"


@dataclass(frozen=True)
class TrainConfig:
    """Parameters for the main training loop"""

    max_episodes: int = 5000
    memory_size: int = 100000
    target_update_frequency: int = 50
    save_frequency: int = 25


@dataclass(frozen=True)
class EnvConfig:
    """Configuration for the game environment and threads"""

    image_size: tuple[int, int] = (84, 84)
    frame_stack: int = 4
    perceptor_fps: int = 60
    thinker_fps: int = 60


@dataclass(frozen=True)
class AppConfig:
    """Root configuration object"""

    agent: AgentConfig = AgentConfig()
    rewards: RewardConfig = RewardConfig()
    training: TrainConfig = TrainConfig()
    env: EnvConfig = EnvConfig()
