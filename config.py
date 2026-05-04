import msgspec


class RewardConfig(msgspec.Struct, frozen=True):
    level_complete_bonus: float = 100.0

    death_penalty: float = -50.0

    depth_reward: float = 2.0
    gem_reward: float = 1.0
    ammo_kill_reward: float = 2.0

    combo_threshold: float = 2.0
    combo_bonus_multiplier: float = 3.0

    step_penalty: float = -0.01
    damage_penalty: float = -2.0

    boundary_penalty_base: float = -0.1
    boundary_penalty_ramp: float = 0.2
    center_pull_coefficient: float = 0.001

    # TODO: consider whether to keep these
    min_reward_clip: float = -100.0
    max_reward_clip: float = 100.0


class AgentConfig(msgspec.Struct, frozen=True):
    learning_rate: float = 0.0001
    gamma: float = 0.99

    epsilon_start: float = 1.0
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.99997

    train_start: int = 500
    batch_size: int = 512

    memory_size: int = 100000
    target_update_tau: float = 0.005

    grad_clip_norm: float = 1.0
    lr_step_size: int = 100000
    replay_flip_probability: float = 0.5


class EnvConfig(msgspec.Struct, frozen=True):
    image_size: tuple[int, int] = (84, 144)
    frame_stack: int = 4
    state_buffer_maxlen: int = 120


# TODO: these will be consts in train.py
MAX_EPISODES: int = 5000
SAVE_FREQUENCY: int = 25
PRETRAINED_MODEL: str = "models/downwell_ai_best.pth"
