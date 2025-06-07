import glob
import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, input_channels=3, num_actions=6):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.action_head = nn.Linear(512, num_actions)
        self.duration_head = nn.Linear(512, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.float() / 255.0
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        actions = self.action_head(x)
        duration = torch.relu(self.duration_head(x)) + 0.02
        return actions, duration


def load_data():
    files = glob.glob("*gameplay*.pkl")
    if not files:
        print("No gameplay files found!")
        return None

    newest_file = max(files, key=os.path.getctime)
    print(f"Loading: {newest_file}")

    try:
        with open(newest_file, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded {len(data)} frames")
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def preprocess_state(state):
    try:
        if len(state.shape) == 3:  # (H, W, C)
            return torch.from_numpy(state).permute(2, 0, 1).float()
        else:
            print(f"Unexpected state shape: {state.shape}")
            return None
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None


def train_from_data():
    print("Training from recorded data")
    print("=" * 30)

    data = load_data()
    if not data: return

    # Filter valid frames
    valid_frames = []
    for frame in data:
        if (isinstance(frame, dict) and
                'state' in frame and
                frame['state'] is not None and
                'action' in frame):
            valid_frames.append(frame)

    print(f"Valid frames: {len(valid_frames)}")

    if len(valid_frames) < 10:
        print("Not enough valid frames!")
        return

    # Prepare training data
    states = []
    actions = []
    durations = []

    for frame in valid_frames:
        state_tensor = preprocess_state(frame['state'])
        if state_tensor is not None:
            states.append(state_tensor)
            actions.append(frame['action'])
            durations.append(frame.get('duration', 0.1))

    if len(states) == 0:
        print("No valid states after preprocessing!")
        return

    print(f"Training on {len(states)} samples")

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(num_actions=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Convert to tensors
    states = torch.stack(states).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    durations = torch.tensor(durations, dtype=torch.float32).to(device)

    print(f"Training on {device}")

    # Training loop
    model.train()
    epochs = 50
    batch_size = 16

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        # Simple batching
        for i in range(0, len(states), batch_size):
            end_idx = min(i + batch_size, len(states))

            batch_states = states[i:end_idx]
            batch_actions = actions[i:end_idx]
            batch_durations = durations[i:end_idx]

            # Forward pass
            pred_actions, pred_durations = model(batch_states)

            # Losses
            action_loss = nn.CrossEntropyLoss()(pred_actions, batch_actions)
            duration_loss = nn.MSELoss()(pred_durations.squeeze(), batch_durations)

            total_loss_batch = action_loss + 0.1 * duration_loss

            # Backward pass
            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()

            total_loss += total_loss_batch.item()
            num_batches += 1

        if epoch % 10 == 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch}/{epochs}: Loss = {avg_loss:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/pretrained_model.pth"

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_path)

    print(f"Model saved to {model_path}")
    print(f"python main.py --load-model {model_path}")


if __name__ == "__main__":
    train_from_data()
