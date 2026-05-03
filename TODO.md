# Downwell.AI - Experiment Roadmap

## Experiment Priority Order

_Quadrant chart plotting all experiments by implementation effort (x-axis) versus expected impact (y-axis). Note: quadrant charts do not support accTitle/accDescr._

```mermaid
quadrantChart
    title Experiment Priority Matrix
    x-axis Low Effort --> High Effort
    y-axis Low Impact --> High Impact
    quadrant-1 Do Now
    quadrant-2 Plan Carefully
    quadrant-3 Deprioritize
    quadrant-4 Quick Wins
    Ammo-delta reward: [0.12, 0.9]
    Combo-delta reward: [0.1, 0.72]
    Horizontal flip aug: [0.15, 0.85]
    Hybrid CNN+numerics: [0.5, 0.82]
    Dueling DQN: [0.52, 0.8]
    N-step returns n=3: [0.45, 0.75]
    Action-weighted eps: [0.12, 0.55]
    gem_high bonus: [0.1, 0.38]
    Ammo-empty penalty: [0.13, 0.45]
    Prioritized ER: [0.8, 0.83]
    Scripted bootstrap: [0.55, 0.7]
    NoisyNets: [0.85, 0.65]
```

## Experiments

| Order | Experiment                                             | File(s) to change                                         | Keep signal                                                                 | Stop signal                                                                   |
| ----- | ------------------------------------------------------ | --------------------------------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| ~~1~~ | ~~Reduce `train_start` 5000->500~~                     | ~~`src/config.py`~~                                       | ~~N/A~~                                                                     | ~~N/A~~                                                                       |
| ~~2~~ | ~~Remove BatchNorm from DQN~~                          | ~~`src/agents/dqn_network.py`~~                           | ~~Loss smoother within 30 eps~~                                             | ~~Loss explodes (revert)~~                                                    |
| ~~3~~ | ~~**[BUG FIX]** Double DQN~~                           | ~~`src/agents/dqn_agent.py`~~                             | ~~Q-values stop growing unboundedly~~                                       | ~~-~~                                                                         |
| ~~4~~ | ~~**[BUG FIX]** Grad clip 10.0->1.0~~                  | ~~`src/agents/dqn_agent.py`, `src/config.py`~~            | ~~Fewer loss spikes~~                                                       | ~~Loss decreases slower (loosen to 5.0)~~                                     |
| ~~5~~ | ~~**[BUG FIX]** Soft target updates (Polyak τ=0.005)~~ | ~~`src/agents/dqn_agent.py`, `src/config.py`, `main.py`~~ | ~~No sudden Q-value jumps; smoother loss~~                                  | ~~Agent learns slower - lower τ to 0.001~~                                    |
| ~~6~~ | ~~**[BUG FIX]** Fix LR schedule~~                      | ~~`src/agents/dqn_agent.py`, `src/config.py`~~            | ~~Loss still decreasing at ep 300~~                                         | ~~Instability late - tighten schedule~~                                       |
| 7     | Ammo-delta enemy-kill reward                           | `src/core/reward_calculator.py`, `src/config.py`          | `max_combo` and `final_gems` rise earlier                                   | No change in `max_combo`/`final_gems` by ep 75; or ammo readings erratic      |
| 8     | Combo-delta reward (replace sustained)                 | `src/core/reward_calculator.py`, `src/config.py`          | `max_combo` grows faster; agent seeks aerial kill sequences                 | `max_combo` no better than baseline by ep 100; loss variance spikes           |
| 9     | Horizontal flip augmentation                           | `src/agents/replay.py`                                    | Same reward trend at fewer real episodes; symmetric Q-values for left/right | Flip creates visible artifacts - check if HUD is asymmetric in cropped frame  |
| 10    | Hybrid input (CNN + numeric state)                     | `src/agents/dqn_network.py`, `src/agents/dqn_agent.py`    | Fewer hp-loss events vs baseline by ep 50                                   | No improvement in avg episode duration by ep 100                              |
| 11    | Dueling DQN                                            | `src/agents/dqn_network.py`                               | Reward variance decreases; `max_ypos_reached` improves earlier              | Advantage stream collapses to near-zero (all Q-values identical)              |
| 12    | N-step returns (n=3)                                   | `src/threaders/thinker.py`                                | Reward variance drops (smoother curve)                                      | Training unstable - loss spikes or reward collapses                           |
| 13    | Prioritized Experience Replay                          | `src/agents/replay.py`                                    | Wider ypos range reached earlier                                            | Training slows without reward improvement by ep 75                            |
| 14    | Action-weighted ε-greedy                               | `src/agents/dqn_agent.py`, `src/config.py`                | Action histogram shifts toward shoot+directional; first combo earlier       | No change in `max_combo`/`final_gems` vs uniform by ep 75                     |
| 15    | gem_high milestone bonus (+20)                         | `src/core/reward_calculator.py`, `src/config.py`          | `final_gems` distribution shifts toward 100                                 | Bonus never fires (agent never near 100 gems in training window)              |
| 16    | Ammo-empty per-step penalty                            | `src/core/reward_calculator.py`, `src/config.py`          | Fewer steps at ammo==0 per episode                                          | Episode duration drops (agent panics when low ammo); or ammo reads unreliable |
| 17    | Scripted bootstrap agent                               | new `src/agents/scripted_agent.py`                        | Fills replay buffer faster than random play                                 | Agent unlearns bootstrap play within 20 eps after switch                      |
| 18    | NoisyNets                                              | `src/agents/dqn_network.py`, `src/agents/dqn_agent.py`    | More diverse exploration near enemies; stable without epsilon tuning        | No diversity in replay; or training instability - revert to ε-greedy          |

## Not Worth Doing

| Experiment                          | Why                                                                                  |
| ----------------------------------- | ------------------------------------------------------------------------------------ |
| Expand frame stack 4->8             | Doubles buffer memory (2.8GB->5.6GB); Downwell's relevant context fits in 267ms      |
| Deeper network (4th conv, wider FC) | Game is simpler than full Atari; current 3-conv already produces 7×7×64 features     |
| Downward velocity reward            | Directly conflicts with combo mechanic - bouncing upward off enemies is correct play |
| Store 60fps transitions             | Consecutive frames share the same action; highly correlated, violates IID assumption |
| Survival bonus                      | Encourages stalling; step penalty already correctly penalizes time                   |

## Architecture

```mermaid
flowchart LR
    accTitle: Downwell.AI Real-Time Data Flow
    accDescr: Three-thread pipeline where PerceptorThread captures game state at 60fps, ThinkerThread runs DQN inference and training at 15fps, and ActorThread sends keyboard inputs back to the game.

    game(["Downwell.exe"])

    subgraph perceptor ["PerceptorThread (60fps)"]
        capture["screenshot +\nmemory read"]
        state["GameState\n(hp, gems, combo,\nxpos, ypos, ammo)"]
        capture --> state
    end

    state_buf[("state_buffer")]

    subgraph thinker ["ThinkerThread (15fps)"]
        dqn["DQNAgent\n(get_action + train)"]
        reward_calc["RewardCalculator"]
        replay_buf[("ReplayBuffer")]
        reward_calc --> dqn
        dqn <--> replay_buf
    end

    action_q[("action_queue")]

    subgraph actor ["ActorThread"]
        kb["keyboard input"]
    end

    game --> capture
    state --> state_buf
    state_buf --> thinker
    dqn --> action_q
    action_q --> kb
    kb --> game

    classDef thread fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef buffer fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12
    classDef external fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class capture,state,dqn,reward_calc thread
    class state_buf,action_q,replay_buf buffer
    class game,kb external
```

## Metrics to Track Per Episode

| Metric             | Meaning                     | Target trend                                |
| ------------------ | --------------------------- | ------------------------------------------- |
| `episode_reward`   | Total shaped reward         | Increasing                                  |
| `duration`         | Seconds alive               | Increasing                                  |
| `max_combo`        | Peak combo reached          | Increasing (signals enemy-bouncing learned) |
| `final_gems`       | Gems at death               | Increasing                                  |
| `max_ypos_reached` | Deepest point reached       | Decreasing (more negative = deeper)         |
| `epsilon`          | Exploration rate            | Decreasing toward 0.1                       |
