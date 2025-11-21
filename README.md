# Downwell.AI

This is a side project I'm working on while learning about AI at university. I'm trying to teach a DQN (Deep Q-Network) agent how to
play Downwell.

## Overview

The way I go about this is by using memory reading (via pymem) to extract game state, screen capture for visual input, and keyboard
automation (pyautogui) to control the game. The AI learns to play through trial and error using deep reinforcement
learning.

**Platform Requirements:** Windows only (uses Windows-specific memory reading and window management)

## Getting Started

### Setup



```bash
# TODO: Figure out how to make these work with uv, it's far better

# Install dependencies
pip install -r requirements.txt

# Install CUDA support for PyTorch
install_cuda.bat
```

### Training the AI

```bash
python main.py
```

**Prerequisites before running:**

- Downwell must be running (downwell.exe)
- Game window must be visible on screen
- The AI will automatically detect and connect to the game process

### Training Output

- Models are saved to `models/` directory:
    - `downwell_ai_best.pth`: Best performing model (highest reward)
    - `downwell_ai_<episode>.pth`: Periodic checkpoints (every 25 episodes by default)
    - `downwell_ai_final_<episode>.pth`: Final model at end of training
- Training history is saved to `training_history.csv` (episode rewards, steps, combos, etc.)

## Architecture

### Three-Thread Pipeline (Orchestrator Pattern)

The core AI system uses a **three-threaded architecture** coordinated by `DownwellAI` (src/core/orchestrator.py):

1. **PerceptorThread** (src/threaders/perceptor.py) - Captures game state at 60 FPS
    - Reads memory values (HP, position, gems, combo, ammo)
    - Captures and preprocesses screenshots
    - Maintains a frame stack (4 frames) for temporal awareness
    - Writes to shared `state_buffer` (deque with thread lock)

2. **ThinkerThread** (src/threaders/thinker.py) - Makes decisions at 60 FPS
    - Reads latest state from `state_buffer`
    - Computes rewards using RewardCalculator
    - Trains the DQN agent (experience replay)
    - Selects actions using epsilon-greedy policy
    - Writes actions to `action_queue`

3. **ActorThread** (src/threaders/actor.py) - Executes actions
    - Reads actions from `action_queue`
    - Manages keyboard state (press/release keys)
    - Uses pyautogui for input simulation

### DQN Agent (src/agents/dqn_agent.py)

- **Network Architecture:** Convolutional neural network (src/agents/dqn_network.py)
    - Input: 4-frame stack (84x84 grayscale images)
    - Output: Q-values for 6 discrete actions
- **Actions:** No-op, Jump, Left, Right, Left+Jump, Right+Jump
- **Training:** Uses target network, experiences and replay buffer
- **Hardware:** Requires NVIDIA GPU (hardcoded to CUDA device)

### Memory Reading (src/environment/mem_extractor.py)

The `Player` class uses pymem to read game state from memory via pointer chains (defined in `utils/game_attributes.py`).
*Memory addresses are Windows-specific and may break with game updates.*

### Reward System (src/core/reward_calculator.py)

- **Primary reward:** Depth (2 per best y-level reached)
- **Bonuses:** Level completion (+100), Gems (+1 each), Combos (+5 each / threshold of 4 needed)
- **Penalties:** Death (-10), Step penalty (-0.01 per step)

Reward weights can be adjusted in `src/config.py` (RewardConfig).

## Configuration

All hyperparameters are in `src/config.py`:

- **AgentConfig**: Learning rate, gamma, epsilon decay, batch size
- **RewardConfig**: Reward weights and clipping
- **TrainConfig**: Episodes, memory size, save frequency
- **EnvConfig**: Image size, frame stack, thread FPS

To modify training behavior, edit these dataclasses (they're frozen, so you'll need to change defaults).

## Common Issues

- **Memory read errors**: Memory offsets in utils/game_attributes.py need updating
- **CUDA out of memory**: Reduce `batch_size` in AgentConfig or `memory_size` in TrainConfig