# JARL (WIP)

Still under active dev. Pivoting for now to develop proprietary on-GPU simulations for improved speed. In its current state, JARL is fit to implement and run a wide variety of algorithms.

JARL is written to be highly modular and allow for rapid prototyping of different RL algorithms.
Eventually, many existing algorithms will be implemented by default within JARL. Their core components will therefore be available to rearrange and refactor into more novel approaches.
JARL utilizes a few core proprietary objects...

### Checkpoint spectator

Watch the two most recently written compatible CARL checkpoints play a 1v1:

```bash
python watch_checkpoints.py
```

Open `http://127.0.0.1:8788`. Drag to orbit, right-drag to pan, and scroll to
zoom. Pass `--blue PATH --orange PATH` to select checkpoints explicitly.

## Runtime Structure

JARL uses an explicit staged runtime:

```text
collect -> store -> sample -> prepare -> optimize -> maintain
```

- `Runner` performs environment interaction.
- Capture components choose optional policy-time fields such as action log
  probabilities, values, and recurrent state.
- `RolloutBuffer` owns ordered consumable on-policy data.
- `ReplayBuffer` owns persistent off-policy data and samples transitions or
  episode-safe windows directly.
- Ordered transforms derive rewards, advantages, returns, and targets.
- Flat and recurrent samplers choose minibatch temporal structure.
- Learners own optimizer ordering, target updates, and scheduler advancement.
- `LearningProgram` optionally sequences coarse stages such as discriminator
  training, learned-reward materialization, and PPO optimization.

The PPO collection boundary is explicit:

```python
captures = (
    LogProbCapture(),
    PolicyVersionCapture(policy),
    ValueCapture(critic),
)
rollout = RolloutBuffer(horizon=128, num_envs=8, device="cuda")
runner = Runner(env, policy, rollout, captures=captures)
```

`ppo.py` and `carl_ppo.py` are complete feed-forward and recurrent examples.
