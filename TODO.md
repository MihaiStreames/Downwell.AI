# TODO

## Phase 1: Port

### Day 1: Scaffold

- [x] `config.py` - port from old
- [x] `src/utils/consts.py` - port from old
- [x] `src/memory/game_ptrs.py` - port from old

### Day 2: Agent

- [x] `src/agent/network.py` - port from old
- [ ] `src/agent/replay.py` - port from old
- [ ] `src/agent/dqn.py` - port from old, fix two things:
  - decouple `GameState` -> accept raw `np.ndarray` in `get_action` / `train`
  - guard `torch.cuda.get_device_name()` (crashes on Linux CPU)

### Day 3: Memory

- [ ] `src/env/memory/pymem_memory.py` - port from old `src/environment/mem_extractor.py`
- [ ] `src/env/memory/null_memory.py` - new Linux stub, same interface, returns zeros/None
- [x] `src/env/memory/__init__.py` - platform factory (`pymem_memory` on win32, `null_memory` on linux)

### Day 4: Capture

- [ ] `src/env/capture/mss_capture.py` - port from old `src/environment/capture.py`
- [ ] `src/env/capture/dxcam_capture.py` - new Windows DXcam wrapper, same interface as mss
- [x] `src/env/capture/__init__.py` - platform factory

### Day 5: Rewards

- [ ] `src/env/reward.py` - port from old `src/core/reward_calculator.py`, decouple from `GameState` -> plain dataclass
- [ ] `src/env/observation.py` - new, builds obs dict from capture + memory outputs (platform-agnostic)

## Phase 2: Study

- [ ] `src/utils/timing.py` - fixed-timestep loop (study rtgym elastic clock pattern)
- [ ] `src/env/input/pynput_input.py` - Linux held-key input
- [ ] `src/env/input/directinput.py` - Windows held-key input (study tmrl diff-based pattern)
- [x] `src/env/input/__init__.py` - platform factory
- [ ] `src/env/game_env.py` - full Gymnasium env, wires capture+input+memory+reward, action buffer in obs
- [ ] `src/train.py` - training entrypoint
