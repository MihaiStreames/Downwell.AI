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


import torch
import numpy as np


from src.consts import FRAME_STACK
from src.consts import IMAGE_HEIGHT
from src.consts import IMAGE_WIDTH
from src.consts import RAM_DIM
from src.consts import ACTION_BUFFER_DIM


class ReplayBuffer:
    def __init__(self, capacity: int, device: torch.device | None = None) -> None:
        self.capacity = capacity
        self.__device = device or torch.device("cpu")

        self.__imgs = np.zeros((capacity, FRAME_STACK, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
        self.__rams = np.zeros((capacity, RAM_DIM), dtype=np.float32)
        self.__acts_buf = np.zeros((capacity, ACTION_BUFFER_DIM), dtype=np.float32)
        self.__actions = np.zeros(capacity, dtype=np.int64)
        self.__rewards = np.zeros(capacity, dtype=np.float32)
        self.__dones = np.zeros(capacity, dtype=bool)

        self.__pos = 0
        self.size = 0

    def sample(
        self, batch_size: int
    ) -> (
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
        | None
    ):

        available = self.size - 1
        if available < batch_size:
            return None

        if self.size < self.capacity:
            idx = np.random.randint(0, available, size=batch_size)
        else:
            exclude = (self.__pos - 1 + self.capacity) % self.capacity
            raw = np.random.randint(0, self.capacity - 1, size=batch_size)
            idx = np.where(raw >= exclude, raw + 1, raw) % self.capacity

        nidx = (idx + 1) % self.capacity

        dev = self.__device

        imgs = torch.from_numpy(self.__imgs[idx]).to(dev, non_blocking=True)
        next_imgs = torch.from_numpy(self.__imgs[nidx]).to(dev, non_blocking=True)

        rams = torch.from_numpy(self.__rams[idx]).to(dev, non_blocking=True)
        next_rams = torch.from_numpy(self.__rams[nidx]).to(dev, non_blocking=True)

        acts = torch.from_numpy(self.__acts_buf[idx]).to(dev, non_blocking=True)
        next_acts = torch.from_numpy(self.__acts_buf[nidx]).to(dev, non_blocking=True)

        actions = torch.from_numpy(self.__actions[idx]).to(dev, non_blocking=True)
        rewards = torch.from_numpy(self.__rewards[idx]).to(dev, non_blocking=True)
        dones = torch.from_numpy(self.__dones[idx]).to(dev, non_blocking=True)

        return (imgs, next_imgs, rams, next_rams, acts, next_acts, actions, rewards, dones)

    def add(self, img: np.ndarray, ram: np.ndarray, acts_buf: np.ndarray, action: int, reward: float, done: bool) -> None:
        i = self.__pos

        self.__imgs[i] = img
        self.__rams[i] = ram
        self.__acts_buf[i] = acts_buf
        self.__actions[i] = action
        self.__rewards[i] = reward
        self.__dones[i] = done
        self.__pos = (i + 1) % self.capacity

        self.size = min(self.size + 1, self.capacity)
