from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """Weights and parameters for the reward function"""
    # --- Core Events ---
    level_complete_bonus: float = 100.0
    death_penalty: float = -100.0
    health_loss_penalty: float = -10.0
    health_gain_reward: float = 20.0

    # --- Guiding Rewards ---
    new_depth_reward: float = 1.5
    gem_base_reward: float = 0.5
    combo_growth_reward: float = 5.0

    # --- Penalties / Time Pressure ---
    base_survival: float = -0.01
    backward_penalty: float = -1.0
    stagnation_penalty: float = -0.05
    stagnation_threshold: int = 240

    # --- Clipping ---
    min_reward_clip: float = -20.0
    max_reward_clip: float = 20.0

    # Disabled for now
    progress_reward: float = 0.0
    combo_base_reward: float = 0.0
    combo_break_penalty: float = 0.0
    high_combo_bonus: float = 0.0
    gem_high_multiplier: float = 1.0


@dataclass(frozen=True)
class AgentConfig:
    """Hyperparameters for the DQN Agent"""
    learning_rate: float = 0.0001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.1
    epsilon_decay: float = 0.999985
    train_start: int = 5000
    batch_size: int = 128
    pretrained_model: str = "models/latest.pth"


@dataclass(frozen=True)
class TrainConfig:
    """Parameters for the main training loop"""
    max_episodes: int = 5000
    memory_size: int = 500000
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
