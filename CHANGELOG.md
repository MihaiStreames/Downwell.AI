# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Project structure: `src/agent/`, `src/env/{capture,input,memory}/` with platform factories
- `src/agent/network.py`: 3-layer CNN with dynamic flattened dim, portrait image size (84x142)
- `src/env/memory/game_ptrs.py`: RAM pointer chains for HP, position, gems, combo, ammo
- `config.py` at project root with all hyperparameters
- Platform factories: win32 vs linux swap for capture, input, memory
- `RESEARCH.md`: real-time RL research findings
- `CHANGELOG.md`
