# Copyright 2023 MihaiStreames, UnderNowhere
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Final


### Process constants ###

PROCESS_NAME: Final[str] = "Downwell.exe"
WINDOW_TITLE: Final[str] = "Downwell"

### Actions ###

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

### Sequences ###

RESET_SEQ: Final[list[str]] = ["esc", "right", "space", "space"]
RETRY_SEQ: Final[list[str]] = ["space", "space", "space"]

### Capture constants ###

CROP_LEFT_RATIO: Final[float] = 0.28
CROP_RIGHT_RATIO: Final[float] = 0.72

IMAGE_WIDTH: Final[int] = 84
IMAGE_HEIGHT: Final[int] = 142
