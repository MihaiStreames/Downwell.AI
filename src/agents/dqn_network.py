import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812


class DQN(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def __init__(self, input_channels=4, num_actions=6):
        super().__init__()

        self._conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self._conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self._conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self._fc1 = nn.Linear(7 * 7 * 64, 512)
        self._fc2 = nn.Linear(512, 256)
        self._action_head = nn.Linear(256, num_actions)

        self._dropout = nn.Dropout(0.1)

        self._initialize_weights()

    def forward(self, x):
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
