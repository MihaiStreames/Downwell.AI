from dataclasses import dataclass
from dataclasses import field


@dataclass
class Config:
    # reward weights
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

    # agent hyperparameters
    learning_rate: float = 0.0001
    gamma: float = 0.9997
    epsilon_start: float = 1.0
    epsilon_min: float = 0.1
    epsilon_decay: float = 0.999985
    train_start: int = 5000
    batch_size: int = 512
    pretrained_model: str = "models/downwell_ai_best.pth"

    # training loop
    max_episodes: int = 5000
    memory_size: int = 100000
    target_update_frequency: int = 50
    save_frequency: int = 25

    # environment / threading
    image_size: tuple[int, int] = field(default_factory=lambda: (84, 84))
    frame_stack: int = 4
    perceptor_fps: int = 60
    thinker_fps: int = 15
