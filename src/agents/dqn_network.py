import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Deep Q-Network for action-value function approximation.

    Convolutional neural network that processes stacked game frames and outputs
    Q-values for each possible action. Uses batch normalization and dropout for
    training stability.

    Parameters
    ----------
    input_channels : int, optional
        Number of stacked input frames (default: 4).
    num_actions : int, optional
        Number of possible actions (default: 6).

    Attributes
    ----------
    conv1, conv2, conv3 : nn.Conv2d
        Convolutional layers for visual feature extraction.
    bn1, bn2, bn3 : nn.BatchNorm2d
        Batch normalization layers for training stability.
    fc1, fc2 : nn.Linear
        Fully connected layers for value computation.
    action_head : nn.Linear
        Output layer producing Q-values for each action.
    dropout : nn.Dropout
        Dropout layer for regularization.
    """

    def __init__(self, input_channels=4, num_actions=6):
        super().__init__()

        # Convolutional layers for image processing
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Batch normalization for stability
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)

        # Processing
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.action_head = nn.Linear(256, num_actions)

        self.dropout = nn.Dropout(0.1)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Kaiming normal initialization.

        Uses He initialization for convolutional and linear layers to prevent
        vanishing/exploding gradients with ReLU activations.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, height, width).
            Can be uint8 (0-255) or float (0-1).

        Returns
        -------
        torch.Tensor
            Q-values for each action, shape (batch, num_actions).
        """
        # Normalize input to [0, 1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        # Conv layers for visual processing
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flatten
        x = x.reshape(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        actions = self.action_head(x)

        return actions
