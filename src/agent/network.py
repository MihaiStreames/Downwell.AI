import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from src.utils.consts import CONV1_CHANNELS
from src.utils.consts import CONV1_KERNEL
from src.utils.consts import CONV1_STRIDE
from src.utils.consts import CONV2_CHANNELS
from src.utils.consts import CONV2_KERNEL
from src.utils.consts import CONV2_STRIDE
from src.utils.consts import CONV3_CHANNELS
from src.utils.consts import CONV3_KERNEL
from src.utils.consts import CONV3_STRIDE
from src.utils.consts import DROPOUT_RATE
from src.utils.consts import FC1_DIM
from src.utils.consts import FC2_DIM
from src.utils.consts import IMAGE_H
from src.utils.consts import IMAGE_W
from src.utils.consts import NUM_ACTIONS


def _conv_out_size(size: int, kernel: int, stride: int, padding: int = 0) -> int:
    return (size - kernel + 2 * padding) // stride + 1


class DQN(nn.Module):
    """TODO: Add docstring."""

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def __init__(
        self,
        input_channels: int = 4,
        num_actions: int = NUM_ACTIONS,
        image_w: int = IMAGE_W,
        image_h: int = IMAGE_H,
    ) -> None:
        super().__init__()

        self._conv1 = nn.Conv2d(input_channels, CONV1_CHANNELS, kernel_size=CONV1_KERNEL, stride=CONV1_STRIDE)
        self._conv2 = nn.Conv2d(CONV1_CHANNELS, CONV2_CHANNELS, kernel_size=CONV2_KERNEL, stride=CONV2_STRIDE)
        self._conv3 = nn.Conv2d(CONV2_CHANNELS, CONV3_CHANNELS, kernel_size=CONV3_KERNEL, stride=CONV3_STRIDE)

        out_w = _conv_out_size(_conv_out_size(_conv_out_size(image_w, CONV1_KERNEL, CONV1_STRIDE), CONV2_KERNEL, CONV2_STRIDE), CONV3_KERNEL, CONV3_STRIDE)
        out_h = _conv_out_size(_conv_out_size(_conv_out_size(image_h, CONV1_KERNEL, CONV1_STRIDE), CONV2_KERNEL, CONV2_STRIDE), CONV3_KERNEL, CONV3_STRIDE)
        flattened = out_w * out_h * CONV3_CHANNELS

        self._fc1 = nn.Linear(flattened, FC1_DIM)
        self._fc2 = nn.Linear(FC1_DIM, FC2_DIM)

        self._action_head = nn.Linear(FC2_DIM, num_actions)
        self._dropout = nn.Dropout(DROPOUT_RATE)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        x = F.relu(self._conv1(x))
        x = F.relu(self._conv2(x))
        x = F.relu(self._conv3(x))

        x = x.reshape(x.size(0), -1)

        x = F.relu(self._fc1(x))
        x = self._dropout(x)
        x = F.relu(self._fc2(x))
        x = self._dropout(x)

        return self._action_head(x)
