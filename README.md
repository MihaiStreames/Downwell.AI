<p align="center">
  <img src="assets/logo.png" alt="Downwell.AI" width="full"/>
</p>

A side project I'm working on while finishing university (given I've now had quite a few AI lessons). The goal is to train a DQN agent to play [Downwell](https://store.steampowered.com/app/360740/Downwell/) with no ROM, no API and no pausing. It watches the live game window and presses keys, the same way I would.

## Status

**Currently being rewritten from scratch** based on a research phase. The original version worked but had  quite a few fundamental timing and architecture problems.

See [RESEARCH.md](RESEARCH.md) for the full findings. *The short version is that real-time games without environment control are a distinct problem class and require a specific approach.*

> **Nothing here is runnable yet. The new architecture is being scaffolded.**

## The problem

Downwell can't be paused or slowed down. The standard RL loop (`observe -> decide -> act -> repeat`) assumes the environment waits while you think. Downwell doesn't. The state keeps changing while the network is doing inference, which breaks the Markov property and causes timing-induced credit assignment errors.

The fix, as described in [Ramstedt & Pal (2019)](https://arxiv.org/abs/1911.04448) and implemented in projects like [tmrl](https://github.com/trackmania-rl/tmrl) and [rtgym](https://github.com/yannbouteiller/rtgym), is to define a fixed wall-clock timestep, run observation/inference/actuation as concurrent threads, and augment the observation with a buffer of recent actions to restore Markovianity under delay. That's the architecture being built here.

## Platform

Training on two machines:

- **Linux laptop** (Intel iGPU, Wayland) - CPU only, used for development
- **Windows desktop** (GTX 1080, CUDA 12.6) - GPU training

Platform-specific modules (screen capture, input, memory reading) are swapped via factory `__init__.py` files. The rest of the codebase is platform-agnostic.

## Details

### Memory reading

The agent can read RAM values directly via `pymem` pointer chains on Windows (HP, position, gems, combo, ammo). Offsets are in [`src/env/memory/game_ptrs.py`](src/env/memory/game_ptrs.py).

> Note: they only work on the Steam build (for now).

## TBD

Action space, algorithm variant, reward shaping, and network architecture are all still open. The original version used Double DQN with 6 discrete actions and a shaped reward - that's the baseline to start from, but nothing is locked in. [RESEARCH.md](RESEARCH.md) has the tradeoffs.

## License

MIT - see [LICENSE](LICENSE) for details.
