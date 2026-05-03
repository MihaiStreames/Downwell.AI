from dataclasses import dataclass
from dataclasses import field
from typing import Final


@dataclass
class Config:
    # reward weights
    level_complete_bonus: Final[float] = 100.0
    death_penalty: Final[float] = -50.0
    depth_reward: Final[float] = 2.0
    gem_reward: Final[float] = 1.0
    ammo_kill_reward: Final[float] = 2.0
    combo_threshold: Final[float] = 2.0
    combo_bonus_multiplier: Final[float] = 3.0
    step_penalty: Final[float] = -0.01
    damage_penalty: Final[float] = -2.0
    min_reward_clip: Final[float] = -100.0
    max_reward_clip: Final[float] = 100.0

    # boundary / center shaping
    boundary_penalty_base: Final[float] = -0.1
    boundary_penalty_ramp: Final[float] = 0.2
    center_pull_coefficient: Final[float] = 0.001

    # agent hyperparameters
    learning_rate: Final[float] = 0.0001
    gamma: Final[float] = 0.99
    epsilon_start: Final[float] = 1.0
    epsilon_min: Final[float] = 0.1
    epsilon_decay: Final[float] = 0.99997
    train_start: Final[int] = 500
    batch_size: Final[int] = 512
    pretrained_model: Final[str] = "models/downwell_ai_best.pth"

    # training loop
    max_episodes: Final[int] = 5000
    memory_size: Final[int] = 100000
    target_update_tau: Final[float] = 0.005
    save_frequency: Final[int] = 25
    grad_clip_norm: Final[float] = 1.0
    lr_step_size: Final[int] = 100000
    replay_flip_probability: Final[float] = 0.5

    # environment / threading
    image_size: Final[tuple[int, int]] = field(default_factory=lambda: (84, 84))
    frame_stack: Final[int] = 4
    perceptor_fps: Final[int] = 60
    thinker_fps: Final[int] = 30
    state_buffer_maxlen: Final[int] = 120
