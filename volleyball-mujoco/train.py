from __future__ import annotations

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from volleyball_env import VolleyballEnv


class SaveVecNormalizeCheckpointCallback(BaseCallback):
    def __init__(self, save_freq: int, save_dir: str, name_prefix: str, verbose: int = 1) -> None:
        super().__init__(verbose=verbose)
        self.save_freq = max(1, save_freq)
        self.save_dir = save_dir
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq != 0:
            return True

        os.makedirs(self.save_dir, exist_ok=True)
        step_tag = self.num_timesteps
        model_path = os.path.join(self.save_dir, f"{self.name_prefix}_{step_tag}_steps")
        vecnorm_path = os.path.join(self.save_dir, f"{self.name_prefix}_{step_tag}_steps_vecnormalize.pkl")

        self.model.save(model_path)
        vec_env = self.model.get_vec_normalize_env()
        if vec_env is not None:
            vec_env.save(vecnorm_path)

        if self.verbose > 0:
            print(f"Saved checkpoint at {step_tag} timesteps to {model_path}.zip")
        return True


def make_env(stage: int, rank: int, base_seed: int):
    def _factory():
        env = VolleyballEnv(
            frame_skip=5,
            render_mode=None,
            opponent_mode="static",
            training_stage=stage,
            max_episode_steps=1000,
        )
        env.reset(seed=base_seed + rank)
        return Monitor(env)

    return _factory


def build_env(stage: int, num_envs: int, base_seed: int):
    env_fns = [make_env(stage, rank=i, base_seed=base_seed) for i in range(num_envs)]
    env = DummyVecEnv(env_fns) if num_envs == 1 else SubprocVecEnv(env_fns, start_method="spawn")
    return VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)


def stage_paths(stage: int) -> tuple[str, str]:
    os.makedirs("models", exist_ok=True)
    return (
        os.path.join("models", f"ppo_stage{stage}"),
        os.path.join("models", f"vecnormalize_stage{stage}.pkl"),
    )


def checkpoint_dir(stage: int) -> str:
    path = os.path.join("models", "checkpoints", f"stage{stage}")
    os.makedirs(path, exist_ok=True)
    return path


def train_stage(
    stage: int,
    total_timesteps: int,
    num_envs: int,
    base_seed: int,
    checkpoint_freq: int,
) -> None:
    rollout_steps = max(256, 2048 // num_envs)
    callback = SaveVecNormalizeCheckpointCallback(
        save_freq=max(1, checkpoint_freq // num_envs),
        save_dir=checkpoint_dir(stage),
        name_prefix=f"ppo_stage{stage}",
    )

    if stage == 1:
        env = build_env(stage=1, num_envs=num_envs, base_seed=base_seed)
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./logs/",
            learning_rate=3e-4,
            n_steps=rollout_steps,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            device="auto",
        )
    else:
        prev_model_path, prev_norm_path = stage_paths(stage - 1)
        env = build_env(stage=stage, num_envs=num_envs, base_seed=base_seed)
        env = VecNormalize.load(prev_norm_path, env)
        env.training = True
        env.norm_reward = True
        model = PPO.load(prev_model_path, env=env, tensorboard_log="./logs/")
        print(
            f"Loaded Stage {stage - 1} model. For the smoothest continuation, keep "
            f"`--num-envs` the same across stages. Current rollout steps per env: {rollout_steps}."
        )

    model.learn(total_timesteps=total_timesteps, callback=callback)
    model_path, norm_path = stage_paths(stage)
    model.save(model_path)
    env.save(norm_path)
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO for staged volleyball learning.")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3], help="Training stage to run.")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Total PPO timesteps for the stage.")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for environment workers.")
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10_000,
        help="Save a checkpoint every N environment timesteps.",
    )
    args = parser.parse_args()
    train_stage(
        stage=args.stage,
        total_timesteps=args.timesteps,
        num_envs=max(1, args.num_envs),
        base_seed=args.seed,
        checkpoint_freq=max(1, args.checkpoint_freq),
    )


if __name__ == "__main__":
    main()
