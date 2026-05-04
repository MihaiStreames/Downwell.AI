from typing import Final


# ---------------------------------
# game process
# ----------------------------------

PROCESS_NAME: Final[str] = "downwell.exe"
WINDOW_TITLE: Final[str] = "Downwell"

# ---------------------------------
# sentinel values
# ---------------------------------

HP_TRANSITION_SENTINEL: Final[float] = 999.0
LEVEL_COMPLETE_YPOS: Final[float] = -100.0
GEM_HIGH_THRESHOLD: Final[float] = 100.0

# ---------------------------------
# playable bounds
# ---------------------------------

WALL_LEFT: Final[float] = 171.0
WALL_RIGHT: Final[float] = 307.0
DANGER_LEFT: Final[float] = 188.0
DANGER_RIGHT: Final[float] = 292.0
CENTER_XPOS: Final[float] = 240.0

# ---------------------------------
# actions
# ---------------------------------

ACTION_NONE: Final[int] = 0
ACTION_JUMP: Final[int] = 1
ACTION_LEFT: Final[int] = 2
ACTION_RIGHT: Final[int] = 3
ACTION_LEFT_JUMP: Final[int] = 4
ACTION_RIGHT_JUMP: Final[int] = 5

ACTION_KEYS: Final[dict[int, set[str]]] = {
    ACTION_NONE: set(),
    ACTION_JUMP: {"space"},
    ACTION_LEFT: {"left"},
    ACTION_RIGHT: {"right"},
    ACTION_LEFT_JUMP: {"left", "space"},
    ACTION_RIGHT_JUMP: {"right", "space"},
}

FLIP_ACTION_MAP: Final[list[int]] = [0, 1, 3, 2, 5, 4]
NUM_ACTIONS: Final[int] = len(ACTION_KEYS)

# ---------------------------------
# screen capture cropping
# ---------------------------------

# window:   760x568
# cropped:  335x568
CROP_LEFT_RATIO: Final[float] = 0.28
CROP_RIGHT_RATIO: Final[float] = 0.72

# ---------------------------------
# CNN architecture
# (coupled to the trained models, do not tweak without retrain!)
# ---------------------------------

IMAGE_W: Final[int] = 84
IMAGE_H: Final[int] = 142

CONV1_CHANNELS: Final[int] = 32
CONV2_CHANNELS: Final[int] = 64
CONV3_CHANNELS: Final[int] = 64

CONV1_KERNEL: Final[int] = 8
CONV2_KERNEL: Final[int] = 4
CONV3_KERNEL: Final[int] = 3

CONV1_STRIDE: Final[int] = 4
CONV2_STRIDE: Final[int] = 2
CONV3_STRIDE: Final[int] = 1

FC1_DIM: Final[int] = 512
FC2_DIM: Final[int] = 256

DROPOUT_RATE: Final[float] = 0.1

ADAM_EPS: Final[float] = 1e-8
LR_SCHEDULER_GAMMA: Final[float] = 0.9
