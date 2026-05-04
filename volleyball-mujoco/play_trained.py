from __future__ import annotations

import argparse
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from volleyball_env import VolleyballEnv


def make_eval_env(stage: int, opponent_mode: str):
    def _factory():
        return VolleyballEnv(
            frame_skip=5,
            render_mode="human",
            opponent_mode=opponent_mode,
            training_stage=stage,
            max_episode_steps=1000,
        )

    return DummyVecEnv([_factory])


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a trained PPO volleyball policy.")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3], help="Model stage to evaluate.")
    parser.add_argument("--opponent-mode", type=str, default="static", choices=["static", "scripted", "random"])
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--model-path", type=str, default=None, help="Optional explicit PPO .zip path.")
    parser.add_argument(
        "--vecnorm-path",
        type=str,
        default=None,
        help="Optional explicit VecNormalize .pkl path for a checkpoint.",
    )
    args = parser.parse_args()

    model_path = args.model_path or os.path.join("models", f"ppo_stage{args.stage}.zip")
    norm_path = args.vecnorm_path or os.path.join("models", f"vecnormalize_stage{args.stage}.pkl")
    env = make_eval_env(args.stage, args.opponent_mode)
    step_dt = env.envs[0].model.opt.timestep * env.envs[0].frame_skip

    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(model_path, env=env)

    for episode in range(args.episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            step_start = time.perf_counter()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            env.render()
            episode_reward += float(reward[0])
            info = infos[0]
            if info.get("hit_up_active"):
                print(
                    f"episode={episode + 1} hit_up active force={info.get('hit_up_force'):.1f} "
                    f"ball_z={info.get('ball_z'):.3f} ball_vz={info.get('ball_vz'):.3f}"
                )
            if info.get("p1_touch") or info.get("crossed_net") or info.get("ball_grounded"):
                print(
                    f"episode={episode + 1} reward={episode_reward:.3f} "
                    f"touch={info.get('p1_touch')} crossed={info.get('crossed_net')} "
                    f"grounded={info.get('ball_grounded')} landed_side={info.get('landed_side')} "
                    f"touch_ball_vz={info.get('touch_ball_vz')} touch_horiz_dist={info.get('touch_horiz_dist')}"
                )
            done = bool(dones[0])
            remaining = step_dt - (time.perf_counter() - step_start)
            if remaining > 0.0:
                time.sleep(remaining)

        print(f"episode={episode + 1} total_reward={episode_reward:.3f}")

    env.close()


if __name__ == "__main__":
    main()
