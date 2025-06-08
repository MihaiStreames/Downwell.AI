import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, input_channels=6, num_actions=8, memory_features=6):
        super(DQN, self).__init__()

        # Convolutional layers for image processing
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Memory features processing (hp, gems, combo, xpos, ypos, ammo)
        self.memory_fc1 = nn.Linear(memory_features, 32)
        self.memory_fc2 = nn.Linear(32, 64)

        # Combined processing: image features + memory features
        self.fc1 = nn.Linear(7 * 7 * 64 + 64, 256)
        self.action_head = nn.Linear(256, num_actions)
        self.duration_head = nn.Linear(256, 1)

        self.relu = nn.ReLU()
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
        x = x.float() / 255.0

        # Conv layers for visual processing
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # Adaptive pooling and flatten
        x = self.adaptive_pool(x)
        image_features = x.view(x.size(0), -1)

        # Process memory features
        normalized_memory = memory_features / torch.tensor([10.0, 1000.0, 50.0, 300.0, 5000.0, 20.0],
                                                           device=memory_features.device)
        memory_processed = self.relu(self.memory_fc1(normalized_memory))
        memory_processed = self.relu(self.memory_fc2(memory_processed))

        # Combine image and memory features
        combined_features = torch.cat([image_features, memory_processed], dim=1)

        # Final layers
        x = self.relu(self.fc1(combined_features))
        x = self.dropout(x)

        # Output action probabilities and duration
        actions = self.action_head(x)
        duration = torch.relu(self.duration_head(x)) * 0.5 + 0.02  # 0.02-0.52s range
        return actions, duration
