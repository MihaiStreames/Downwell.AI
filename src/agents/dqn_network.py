import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_channels=4, num_actions=8, memory_features=6):
        super(DQN, self).__init__()

        # Convolutional layers for image processing
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Batch normalization for stability
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)

        # Memory features processing (hp, gems, combo, xpos, ypos, ammo)
        self.memory_fc1 = nn.Linear(memory_features, 32)
        self.memory_fc2 = nn.Linear(32, 64)

        # Combined processing: image features + memory features
        self.fc1 = nn.Linear(7 * 7 * 64 + 64, 256)
        self.action_head = nn.Linear(256, num_actions)

        self.register_buffer(
            'memory_norm',
            torch.tensor([10.0, 1000.0, 50.0, 300.0, 5000.0, 20.0])
        )

        self.dropout = nn.Dropout(0.1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, memory_features):
        # Normalize input to [0, 1]
        if x.dtype == torch.uint8: x = x.float() / 255.0

        # Conv layers for visual processing
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flatten
        image_features = x.reshape(x.size(0), -1)

        # Process memory features
        normalized_memory = memory_features / self.memory_norm
        memory_processed = F.relu(self.memory_fc1(normalized_memory))
        memory_processed = F.relu(self.memory_fc2(memory_processed))

        # Combine and output
        combined_features = torch.cat([image_features, memory_processed], dim=1)
        x = F.relu(self.fc1(combined_features))
        x = self.dropout(x)
        actions = self.action_head(x)

        return actions
