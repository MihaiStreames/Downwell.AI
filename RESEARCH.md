# Research notes: real-time RL on a live PC game

Notes from researching how to train an RL agent on a game I can't pause, slow down, or hook into. No public Downwell RL project exists, so I had to piece this together from adjacent work - mostly TrackMania projects and a handful of academic papers.

## The short version

The pattern is called **Real-Time RL (RTRL) with a Delayed/Augmented MDP**. The idea: define a fixed wall-clock timestep, append a buffer of the last *N* actions to every observation to restore the Markov property, and run observation/inference/actuation as concurrent threads. This is exactly what `rtgym` and `tmrl` do. The action space should be "commanded held-key state" rather than taps. Input goes through `pydirectinput` or `vgamepad`, not `pyautogui`.

## Academic foundations

The papers behind this, in the order they matter:

**Travnik, Mathewson, Sutton & Pilarski (2018) - "Reactive Reinforcement Learning in Asynchronous Environments"**
The first paper to formally state that the standard MDP breaks for real-time games: the state keeps evolving while the agent is computing. Introduces the asynchronous-environment formulation.

**Ramstedt & Pal (2019) - "Real-Time Reinforcement Learning"** (NeurIPS, arXiv:1911.04448, code: `github.com/rmst/rtrl`)
Defines the **Real-Time MDP (RTMDP)**: the action chosen at step *t* is applied at step *t+1*, not *t*. They prove you can re-express this as a standard turn-based MDP by augmenting the state with the most recent committed action. This is the theoretical basis for the action buffer trick.

**Ramstedt, Bouteiller et al. (2020) - "Reinforcement Learning with Random Delays"** (RLRD/DCAC, arXiv:2010.02966)
Generalizes to random observation + action delays, which is the real situation - timing is jittery in practice. The fix: augment the observation with a buffer of size `⌈max_delay / timestep⌉` of past actions. For off-policy methods, optionally resample trajectory fragments in hindsight.

**Derman, Dalal & Mannor (2021) - "Acting in Delayed Environments with Non-Stationary Markov Policies"** (arXiv:2101.11992)
**Nath et al. (2021) - "Revisiting State Augmentation for Stochastic Delays" (DRDQN)** (arXiv:2108.07555)
DRDQN is the discrete-action version of DCAC - directly applicable to a 6-action Downwell setup.

**Yuan & Mahmood (2022) - "Asynchronous RL for Real-Time Control of Physical Robots"** (arXiv:2203.12759)
Empirical comparison of sequential vs. asynchronous RL loops on hardware that can't pause. Sequential implementations effectively act on stale observations. Confirms the async approach is necessary, not just nice-to-have.

**"Enabling Realtime RL at Scale with Staggered Asynchronous Inference"** (arXiv:2412.14355, 2024)
Benchmarks on Pokémon and Tetris running in real time. Defines the **asynchronous MDP** `⟨S, A, p, r, β⟩` where `β` is a *default behavior* policy applied when the agent hasn't finished computing yet - exactly the situation I'm in when the inference thread is mid-forward-pass. They propose running multiple staggered inference processes so an action is always ready at every clock tick.

**Lee et al. (2024) - "Pausing Policy Learning in Non-stationary RL"** (arXiv:2405.16053)
**Hu et al. (2024) - "State-Novelty Guided Action Persistence" (SNAP)** (arXiv:2409.05433)
Rounding out the recent literature.

The unifying answer from all of these: **the MDP step is wall-clock-defined, not game-loop-defined**. Replace `env.step()` with an elastic clock that fires every ΔT seconds, and make the state Markov by appending the action buffer.

## Reference projects

### TrackMania - tmrl (`github.com/trackmania-rl/tmrl`)

The gold standard for my situation. Built by Yann Bouteiller (author of the RLRD paper above). Key decisions:

- Built on **rtgym** (`github.com/yannbouteiller/rtgym`), a thin Gymnasium wrapper that runs a background thread enforcing a fixed nominal timestep ("elastically constrained"). When inference is fast, rtgym sleeps until the next tick; when slow, the timestep stretches and fires a warning. Achievable rate: ~50 Hz on Windows (limited by `Sleep()` granularity ≈16 ms), ~500 Hz on Linux.
- Automatically augments the observation with **the last 4 actions** - the RLRD action buffer, operationalised.
- Default action is sent on `reset()` because the game doesn't pause. There's a `wait()` API but typical usage just keeps the loop running.
- Original TMNF version doesn't read game memory - speed was estimated by 1-NN regression on screenshots. TM2020 later added OpenPlanet API for memory access. Same pattern as my RAM extraction.
- Algorithm: **SAC** and **REDQ** via `vgamepad` (`github.com/yannbouteiller/vgamepad`), a virtual XBox360/DS4 device. Not pyautogui.

### TrackMania - Linesight (`github.com/Linesight-RL/linesight`)

State-of-the-art TM AI - beat 10/12 official TM2020 campaign world records by mid-2024. Blog: `linesight-rl.github.io/linesight`, deep dive: `hallofdreams.org/posts/trackmania-1/`.

Relevant differences from tmrl:

- Uses **IQN (Implicit Quantile Networks)**, a distributional DQN variant, with discrete actions. Directly applicable to Downwell.
- 12-action discrete space (key-state combinations), each held for one timestep.
- Can speed up the game via TMInterface - I can't do this with Downwell. Means I have to train in real time and use more sample-efficient algorithms (Rainbow-DQN, IQN, REDQ).
- Uses **DXcam** (`github.com/ra1nty/DXcam`) for ~240 Hz capture via the Windows Desktop Duplication API. DXcam was written specifically for this project because nothing else was fast enough.

### Other relevant projects

**TMAI** (`github.com/LouisDeOliveira/TMAI`) - Win32 API input, holds keys while the action vector says they should be held. Their `control_keyboard.py` is worth reading for the diff-based key-hold pattern.

**SoulsGym** (`github.com/amacati/SoulsGym`, blog: `amacati.github.io/posts/2023/05/soulsgym/`) - RL for Dark Souls III bosses. Uses `pymem`-style RAM reading. They DLL-inject a speed hack that reroutes Windows performance counters to a custom timer they control - they say this is the only way to make standard turn-based RL work cleanly. If I won't do that, I need the real-time approach.

**PokeRL / PyBoy projects** (Bas de Haan's blog, `arxiv.org/abs/2604.10812`, Peter Whidden's 2023 project) - have emulator control but show the canonical RAM-read reward pattern: every reward variable maps to specific memory addresses found with Cheat Engine. That's exactly my setup.

**Geometry Dash DQN** (`github.com/SynvexAI/GDind`, Stanford CS231N report `cs231n.stanford.edu/reports/2017/pdfs/605.pdf`) - closest in spirit to Downwell: tight-timing 2D platformer, binary action, screen capture only. Both GD projects struggled with timing-induced credit-assignment errors. Validation that this is the hard part.

**Heroic Magic Duel** (Warchalski et al., arXiv:2002.06290) - PPO + self-play on a real-time mobile game. Shows PPO is a viable alternative to DQN for real-time.

## Action space

Three patterns:

**(a) Commanded held-state** - what tmrl and Linesight do, and what I'm going with. The action is an index encoding which buttons are currently pressed. The acting thread holds those keys and only sends key-up/key-down *deltas* when the action changes. No separate "tap" concept - a tap is just held for one timestep and then released the next. For Downwell: `{none, jump, left, right, left+jump, right+jump}` = 6 held configurations per timestep. At ΔT=50ms (3 game frames), the agent can tap by holding one step and releasing the next.

**(b) Frame-skip / action repeat** - pick action, repeat for *k* steps. Standard in Atari DQN. Reduces control frequency but improves credit assignment. See Kalyanakrishnan et al. "An Analysis of Frame-skipping in RL" (arXiv:2102.03718) and Daniel Seita's blog post (`danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games`).

**(c) Action + duration (FiGAR / dynamic frame-skip)** - the agent picks both an action *and* a repeat count. Papers: "Dynamic Frame skip Deep Q Network" (arXiv:1605.05365), Sharma et al. "Learning to Repeat: Fine Grained Action Repetition for Deep RL" (FiGAR), Metelli et al. "Control Frequency Adaptation via Action Persistence" (PFQI, arXiv:2002.06836), Hu et al. SNAP (arXiv:2409.05433). Overkill for Downwell; (a) at 50-100ms is the right call.

The implementation: `current_held` set in the acting thread. On each new action, diff against the required set and send only the changes. This is tmrl's `apply_control` / `keyres` pattern in `tmrl/custom/tm/utils/control_keyboard.py`.

## Input

`pyautogui` has a built-in 0.1s PAUSE after every call and uses the legacy `keybd_event` API that many DirectX games ignore. Replacements, in order of preference:

| Library | Mechanism | Pros | Cons |
|---|---|---|---|
| **pydirectinput-rgx** | `SendInput()` + DirectInput scancodes | Drop-in pyautogui API, no PAUSE, works with most DirectX games, pure Python | Some anti-cheats detect; blocked when window not focused |
| **vgamepad** | ViGEmBus virtual XBox360/DS4 driver | Cleanest held-button semantics, analog support, what tmrl uses | Needs ViGEmBus driver installed, gamepad must be mapped in-game |
| **pyinterception** | Kernel-level driver | Lowest latency, `LLKHF_INJECTED=False` (looks like real hardware), global hotkeys | Needs Interception driver, some anti-cheats refuse to run with it |
| **win32api / ctypes** | Raw Win32 | Zero deps, full scancode control | Boilerplate |
| **pyautogui** | `keybd_event` (legacy) | Easy | 0.1s PAUSE, legacy API, often ignored |

For Downwell (Game Maker Studio, standard keyboard input): `pydirectinput-rgx` is the simplest fix on Windows. Downwell also has first-class gamepad support so `vgamepad` is an option if I want to go that route.

For Linux/Wayland: `pynput`. No `Sleep()` granularity problem on Linux so timing is actually better than Windows.

For screen capture: **DXcam** (`github.com/ra1nty/DXcam`) or **BetterCam** (`github.com/RootKit-Org/BetterCam`) on Windows - 5-10x faster than mss/PIL/pyautogui via the Desktop Duplication API, 60-240 Hz. On Linux: fastgrab or a Wayland screencopy approach (mss doesn't work under Wayland).

## Timing

Three approaches:

**(i) Fixed wall-clock with elastic constraint (rtgym pattern)** - pick ΔT (e.g. 50ms = 20 Hz). A clock thread fires every ΔT. On each tick: apply the queued action, capture observation, hand to inference thread. If inference is slow, the timestep stretches and the next one compensates. The agent's observation includes the last N actions so the policy can reason about in-flight commands - this is what makes the augmented MDP Markov.

**(ii) Frame-counter-based** - if Downwell has a monotonically increasing frame counter or timer in RAM, lock the decision loop to it (decide every k frames). Eliminates wall-clock jitter. Worth looking for: a 32-bit value incrementing by 1 per game frame is the free synchronization signal.

**(iii) Vsync-tick capture (advanced)** - DXcam's `target_fps` + Windows `CREATE_WAITABLE_TIMER_HIGH_RESOLUTION` gives ±1ms accuracy.

Concrete setup for 60 FPS Downwell:

- Decision frequency: 20-30 Hz (ΔT = 33-50ms = 2-3 game frames)
- Observation thread at 60 FPS sampling RAM; on each decision tick use the latest cached observation
- Acting thread holds current commanded action; only sends key deltas on change, nothing between
- Observation to network: `[ram_features..., one_hot(prev_action_t-1), one_hot(prev_action_t-2), one_hot(prev_action_t-3)]`
- With pydirectinput latency ≈1ms and ΔT=50ms, buffer length N=3-4 is enough. Formula from RLRD Appendix: `N ≥ ⌈max_total_delay / ΔT⌉`

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    rtgym-style elastic clock                │
│                       (ticks every ΔT)                      │
└─────────────────────────────────────────────────────────────┘
       │                                              │
       │ tick                                         │ tick
       ▼                                              ▼
┌──────────────┐  obs queue  ┌──────────────┐  act queue ┌──────────────┐
│ Observation  │────────────▶│  Inference   │───────────▶│   Acting     │
│  Thread      │             │  Thread      │            │   Thread     │
│              │             │              │            │              │
│ - pymem read │             │ - DQN/IQN    │            │ - vgamepad / │
│  (ram->feat) │             │   forward    │            │   pydirect-  │
│ - DXcam grab │             │ - argmax + ε │            │   input      │
│ - 60 FPS     │             │ - 30 Hz      │            │ - diff held  │
└──────────────┘             └──────────────┘            │   keys; only │
                                    │                    │   send delta │
                                    │ store transition   └──────────────┘
                                    ▼
                              ┌──────────────┐
                              │ Replay buffer│
                              │ (s,a,r,s')   │
                              │ s incl. last │
                              │ N actions    │
                              └──────────────┘
                                    │
                                    ▼
                              ┌──────────────┐
                              │ Learner      │
                              │ Thread       │
                              │ (separate)   │
                              └──────────────┘
```

Stored transition: `(s_t, a_t, r_t, s_{t+1})` where `s_t` includes RAM features plus the action buffer.

## What to read, in priority order

1. **rtgym** (`github.com/yannbouteiller/rtgym`) - README + `tuto/tuto.py`. The canonical implementation. ~30 min.
2. **tmrl `tm_gym_interfaces.py`** (`github.com/trackmania-rl/tmrl/blob/master/tmrl/custom/tm/tm_gym_interfaces.py`) - how `RealTimeGymInterface` is subclassed for a real game. `send_control`, `get_obs_rew_terminated_info`, `reset`. Also `control_keyboard.py` for the diff-based key-hold pattern.
3. **Ramstedt & Pal, arXiv:1911.04448** - §2 (RTMDP definition) and §3 (why standard SAC is suboptimal, what to do).
4. **Bouteiller et al., arXiv:2010.02966** - §3 and the Appendix on sizing the action buffer. The exact formula for N given latencies.
5. **Linesight** (`github.com/Linesight-RL/linesight`) - `trackmania_rl/` directory: IQN network, discrete action set, rollout workers feeding a central learner. Closest discrete-action analogue to Downwell.
6. **SoulsGym blog** (`amacati.github.io/posts/2023/05/soulsgym/`) - the speed-hack DLL injection alternative, different perspective on the same problem.
7. **Hall of Impossible Dreams Trackmania series** (`hallofdreams.org/posts/trackmania-1/` etc.) - best long-form explanation of the engineering tradeoffs: screen capture choice, OS timing, plugin vs. memory reading.
8. **HN thread** (`news.ycombinator.com/item?id=40853834`) - direct comments from TMInterface developer Donadigo and Linesight maintainer on what RL projects actually need.

## Caveats

**rtgym timing on Windows:** `time.sleep()` granularity is ~16ms, so 30-50ms timesteps are reliable but anything faster will jitter. Python 3.11+ helps with `CreateWaitableTimerExW HIGH_RESOLUTION` flag (same trick DXcam uses). Linux sleep is ~1ms so the laptop is actually better for timing than the desktop.

**Action buffer length:** too short = still non-Markov; too long = input dimension blows up. Formula: `N = ⌈(max_observation_delay + max_action_delay) / ΔT⌉ + 1`. With pydirectinput, delays drop to a few ms each, so N=2-4 is enough at ΔT=33-50ms.

**Algorithm choice:** Linesight uses IQN (distributional DQN); tmrl uses SAC (continuous). DQN/Rainbow/IQN fit the discrete 6-action space. PPO is also viable - most real-time game papers (Heroic, Naruto Mobile, Geometry Dash) use it for stability. Plain DQN often struggles in real-time without delay-corrected target computation - DRDQN (Nath et al.) or IQN with action-buffer augmentation are better starting points.

**CPU training (Linux laptop):** Rainbow DQN is overkill without a GPU. Plain DQN or DQN+PER to start. PPO is worth considering - on-policy means no replay buffer RAM overhead. Replay buffer at 50k-100k transitions max. A small CNN (3 conv + 2 FC) is the ceiling without a GPU.

**No public Downwell RL project exists** (as of mid-2026). Closest analogues: Geometry Dash DQN projects, Pokémon-Red PPO projects (both use emulator control). The TrackMania ecosystem is the only mature reference for the "live PC game, no env control, real-time" pattern.

**Speculative claims in blog posts:** hallofdreams.org and similar describe planned/in-progress work in future tense. Treat target numbers as aspirational. Verified results: Linesight beat 10/12 official TM2020 campaign world records (May 2024, on YouTube), tmrl reached ~45.5s on tmrl-test track vs ~32s human target (June 2022), SoulsGym reached ~45% win rate against Iudex Gundyr (2023).
