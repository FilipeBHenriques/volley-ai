# VolleyAI MuJoCo RL Setup

This project includes a staged Gymnasium environment and PPO training flow for a 1v1 MuJoCo volleyball simulator. The first milestones are:

1. Player 1 moves toward the ball.
2. Player 1 touches the ball reliably.
3. Player 1 starts lifting the ball upward.
4. Player 1 occasionally sends the ball over the net.

## Files

```text
volleyball-mujoco/
|-- assets/volleyball.xml
|-- volleyball_core.py
|-- volleyball_env.py
|-- train.py
|-- play_trained.py
|-- random_test.py
|-- requirements.txt
`-- README.md
```

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Environment

`VolleyballEnv` exposes:

- action space `Box(-1, 1, shape=(3,))`
- action meaning `[lateral_y, depth_x, hit_up]`
- observation size `18`
- opponent modes `static`, `scripted`, `random`
- training stages `1` through `4`

Recommended training order:

1. Stage 1 with `opponent_mode="static"`
2. Stage 2 with `opponent_mode="static"`
3. Stage 3 with `opponent_mode="static"`
4. Only try Stage 4 after Stage 3 works visually

## Train

Stage 1:

```bash
python train.py --stage 1 --timesteps 200000 --num-envs 4
```

With automatic checkpoints every 10k steps:

```bash
python train.py --stage 1 --timesteps 200000 --num-envs 4 --checkpoint-freq 10000
```

Stage 2:

```bash
python train.py --stage 2 --timesteps 200000 --num-envs 4
```

Stage 3:

```bash
python train.py --stage 3 --timesteps 200000 --num-envs 4
```

Speed tip:

- `--num-envs 4` is a good default on a typical desktop CPU
- try `--num-envs 8` if you have many cores and enough RAM
- keep the same `--num-envs` across Stage 1, 2, and 3 if you continue from saved checkpoints
- for quick debugging, use `--num-envs 1 --timesteps 10000`

Saved artifacts:

- `models/ppo_stageN.zip`
- `models/vecnormalize_stageN.pkl`
- `models/checkpoints/stageN/ppo_stageN_<steps>_steps.zip`
- `models/checkpoints/stageN/ppo_stageN_<steps>_steps_vecnormalize.pkl`
- TensorBoard logs in `logs/`

## Evaluate

```bash
python play_trained.py --stage 1 --opponent-mode static
```

Evaluate a specific checkpoint:

```bash
python play_trained.py --stage 1 --opponent-mode static ^
  --model-path models/checkpoints/stage1/ppo_stage1_10000_steps.zip ^
  --vecnorm-path models/checkpoints/stage1/ppo_stage1_10000_steps_vecnormalize.pkl
```

Later, once Stage 3 starts working:

```bash
python play_trained.py --stage 3 --opponent-mode scripted
```

## Random smoke test

```bash
python random_test.py
```

This checks:

- observations remain finite
- rewards remain finite
- episodes terminate or truncate
- the environment survives random actions

## Why checkpoints help

Checkpoint saving is usually worth it because it lets you:

- stop training anytime with much lower risk of losing progress
- compare behavior at `10k`, `20k`, `50k`, and later
- inspect whether Player 1 is actually moving toward the ball yet
- recover more easily if a long run crashes

The main tradeoff is a bit more disk usage and a small amount of save overhead, but for this project that cost is usually minor compared with the visibility and safety you gain.

## Reward stages

Stage 1 focuses on approach and touch:

- alive bonus while the ball stays up
- horizontal distance penalty
- touch reward
- ground-contact penalty

Stage 2 adds lift shaping:

- stronger touch reward
- upward ball velocity reward after touch
- bonus for getting the ball above a useful height

Stage 3 adds net progress:

- reward for sending the ball over the net above `2.43`
- reward for opponent-side landings
- penalty for own-side landings

## Practical checks

After Stage 1:

- touches per episode should increase
- Player 1 should visibly move toward the ball

If Stage 1 stalls:

- reduce ball randomness
- slow the ball down
- strengthen distance shaping
- increase `MOVE_FORCE` slightly

After Stage 2:

- check whether upward press behavior appears near the ball
- inspect whether ball vertical velocity improves after contact

If Stage 2 stalls:

- start the ball closer
- increase upward-velocity reward
- lower jump timing difficulty

After Stage 3:

- first look for occasional successful crossings
- only later optimize landing quality
