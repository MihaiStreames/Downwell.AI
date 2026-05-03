<p align="center">
  <img src="assets/logo.png" alt="Downwell.AI" width="full"/>
</p>

A side project I'm working on while learning about AI at university. The goal is to teach a DQN agent how to play [Downwell](https://store.steampowered.com/app/360740/Downwell/) - I wanted something harder than CartPole and more fun than Atari benchmarks.

## Overview

The AI watches the game (screen capture + memory reads via `pymem`) and presses keys with `pyautogui`. There's no API or ROM hooks, it simply plays the real game in a real window, the same way I would. That makes everything messier than a clean Gym env, but also more interesting: the agent has to deal with frame drops, transition screens, and a game that wasn't designed to be poked at.

The brain is a fairly standard Double-DQN with frame stacking and Polyak-averaged target updates. Most of the actual work has been in the *non-AI* parts: getting state cleanly, shaping rewards that don't reward the wrong thing, and keeping three threads in sync without losing too much information.

**Platform Requirements:** Windows only (uses Windows-specific memory reading and window management).

## Running it

You'll need [uv](https://docs.astral.sh/uv/), Python 3.10+, an NVIDIA GPU with CUDA 12.6, and a copy of Downwell.

```bash
uv sync --extra windows
uv run python -c "import torch; print(torch.cuda.is_available())"
```

Then start Downwell, make sure the window is visible, and:

```bash
uv run python main.py
```

Models drop into `models/` (`downwell_ai_best.pth` is the highest-reward one, plus periodic checkpoints every 25 episodes). Episode stats go to `training_history.csv`. To see the curves:

```bash
uv run python plotter.py
```

If you're on Linux and just want to read the code, `uv sync` (without the windows extra) is enough.

## Architecture

### Three-thread pipeline

The core AI system uses a three-threaded architecture coordinated by `DownwellAI` ([orchestrator.py](src/core/orchestrator.py)):

1. **PerceptorThread** ([perceptor.py](src/threaders/perceptor.py)) - captures game state at 60 FPS
   - Reads memory values (HP, position, gems, combo, ammo)
   - Captures and preprocesses screenshots
   - Maintains a frame stack (4 frames) for temporal awareness
   - Writes to a shared `state_buffer` (deque with thread lock)

2. **ThinkerThread** ([thinker.py](src/threaders/thinker.py)) - makes decisions at 30 FPS
   - Reads latest state from `state_buffer`
   - Computes rewards using `RewardCalculator`
   - Trains the DQN agent (experience replay)
   - Selects actions using epsilon-greedy policy
   - Drains stale actions from the queue before pushing a new one, so the actor never executes a decision that's two frames out of date
   - Writes actions to `action_queue`

3. **ActorThread** ([actor.py](src/threaders/actor.py)) - executes actions
   - Reads actions from `action_queue`
   - Manages keyboard state (press/release keys)
   - Uses pyautogui for input simulation

### DQN Agent

CNN over a 4-frame stack of 84x84 grayscale images ([dqn_network.py](src/agents/dqn_network.py)), six discrete actions: no-op, jump, left, right, left+jump, right+jump. Replay buffer holds 100k transitions. Target network updates with Polyak averaging (`tau=0.005`) instead of hard copies - smoother, less destabilizing. Standard ε-greedy with slow decay.

Nothing exotic. The interesting stuff is in the rewards.

### Memory Reading

Pointer chains in [game_attributes.py](src/utils/game_attributes.py). `Player.get_value()` walks them and returns whatever the type says. Some attributes have multiple base addresses to try since some would occasionally get relocated.

These offsets *will* break if the game updates. They're written for the current Steam build - it won't work on the standalone version since that doesn't have the Steam hooks. I'd like to look into that later as I want to run multiple instances.

### Reward shaping (the part I keep rewriting)

Naive reward shaping in this game is a trap. The big one I'm still fighting:

**Room farming.** The agent gets stuck in side rooms / shops / power-up rooms and accumulates reward by bouncing around. The first version rewarded absolute depth, so revisiting depths in a shop just kept paying out. I switched to per-level depth *delta* (only new max-depth progress within a level counts) and added an out-of-bounds penalty for being near the side walls. Neither was enough on its own, the agent still doesn't really learn to backtrack out of a room it walked into. This is an open problem.

**Wall sticking.** Related to the room thing, a separate failure mode. Agent gets pinned at a left/right wall and jumps in place. Currently mitigating with action filtering (strips left/right inputs when xpos is past a threshold) and a danger-zone penalty before the wall, plus a small center-attraction reward. Still not solved, the next plan is a stuck-detector that forces a bounce when xpos doesn't move for N frames.

The current reward weights:

- **Bonuses:** level completion (+100), depth progress (+2 per unit deeper than max delta this level), gems (+1 each), ammo-kill (+2 per ammo unit fired while in combo), combo (sustained combo over a threshold)
- **Penalties:** death (-50), step (-0.01), damage (-2 per HP lost), boundary danger zone, center pull

Rewards clip to `[-100, 100]` so a single bad transition can't poison the buffer. All weights in [config.py](src/config.py) (single `Config` dataclass).

## Configuration

All hyperparameters live in [config.py](src/config.py), grouped by section:

- **Reward weights**: bonuses, penalties, clipping bounds
- **Agent hyperparameters**: learning rate, gamma, epsilon schedule, batch size, pretrained model path
- **Training loop**: max episodes, memory size, target update tau (Polyak), save frequency, gradient clipping, LR step size
- **Environment / threading**: image size, frame stack depth, perceptor/thinker FPS

## Things that bit me

- **Transition frames.** Between levels, xpos and hp become unreadable. Originally the agent saw `hp=None` and panicked. Now the perceptor returns a sentinel `hp=999.0` during transitions and the thinker / reward calc ignore those frames entirely.
- **Pointer-chain log spam.** Failed reads at 60fps flood the logs during transitions. Dropped those messages to TRACE level.

## Common Issues

- **Memory read errors:** offsets in [game_attributes.py](src/utils/game_attributes.py) may need updating after game patches.
- **CUDA out of memory:** reduce `batch_size` or `memory_size` in [config.py](src/config.py).
- **"Downwell window not found":** make sure the game is running and the window is visible.
- **Import errors on Linux:** expected, the Windows-only dependencies aren't needed for code editing.

## Contributing

This is a learning project, but feel free to open issues or PRs if you find bugs or have suggestions!

## License

MIT - see [LICENSE](LICENSE) for details.
