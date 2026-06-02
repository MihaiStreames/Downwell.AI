# Copyright 2023 MihaiStreames
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


PLAYER_PTR: dict[str, dict[str, object]] = {
    "ypos": {"base": 0x0005220C, "offsets": [0x360], "type": "float"},
    "xpos": {
        "bases": [0x00534288, 0x005479FC],
        "offsets": [
            [0x100, 0x8C0, 0x10, 0x84, 0x44, 0x8, 0xB0],
            [0x240, 0x4F0, 0x10, 0x84, 0x44, 0x8, 0xB0],
        ],
        "type": "float",
    },
    "hp": {
        "base": 0x004A5E50,
        "offsets": [0x708, 0xC, 0x24, 0x10, 0x9C0, 0x390],
        "type": "double",
    },
    "gems": {
        "base": 0x00757BF0,
        "offsets": [0x24, 0x10, 0x330, 0xE0, 0x50, 0x9A8, 0x350],
        "type": "double",
    },
    "ammo": {
        "bases": [0x00757C80, 0x00757BF8, 0x00757978],
        "offsets": [
            [0x88, 0x160, 0x50, 0x804, 0x150],
            [0x24, 0x10, 0x4D4, 0x160, 0x50, 0xE4C, 0x660],
            [0x324, 0xE0, 0x8, 0x8, 0x50, 0xF78, 0xA0],
        ],
        "type": "double",
    },
    "gem_high": {
        "base": 0x004A5E50,
        "offsets": [0x708, 0xC, 0x24, 0x10, 0x9E4, 0x480],
        "type": "double",
    },
    "combo": {
        "base": 0x00757C80,
        "offsets": [0x168, 0x160, 0x8, 0x8, 0x50, 0x108, 0x2C0],
        "type": "double",
    },
}
