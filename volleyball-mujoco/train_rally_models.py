from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import mujoco

from play_match import (
    NET_HEIGHT,
    ScriptedPolicy,
    _actual_action,
    _apply_volleyball_touch,
    _contacted,
    _observation,
    _reachable,
    _serve,
    rally_features,
)
from volleyball_core import _build_indices, apply_action


class RallyNet(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _sample_obs(rng: np.random.Generator, n: int) -> np.ndarray:
    player_x = rng.uniform(0.35, 7.8, size=n)
    player_y = rng.uniform(-3.6, 3.6, size=n)
    player_z = rng.uniform(0.55, 1.4, size=n)
    ball_x = rng.uniform(-1.0, 7.8, size=n)
    ball_y = rng.uniform(-3.8, 3.8, size=n)
    ball_z = rng.uniform(0.45, 4.2, size=n)
    ball_vx = rng.uniform(-8.0, 8.0, size=n)
    ball_vy = rng.uniform(-1.0, 1.0, size=n)
    ball_vz = rng.uniform(-7.0, 5.5, size=n)

    near_count = n // 3
    near_idx = rng.choice(n, size=near_count, replace=False)
    ball_x[near_idx] = player_x[near_idx] + rng.uniform(-0.75, 0.75, size=near_count)
    ball_y[near_idx] = player_y[near_idx] + rng.uniform(-0.55, 0.55, size=near_count)
    ball_z[near_idx] = rng.uniform(0.8, 3.05, size=near_count)
    ball_vz[near_idx] = rng.uniform(-6.0, 1.2, size=near_count)

    obs = np.zeros((n, 18), dtype=np.float32)
    obs[:, 0] = player_x
    obs[:, 1] = player_y
    obs[:, 2] = player_z
    obs[:, 6] = ball_x
    obs[:, 7] = ball_y
    obs[:, 8] = ball_z
    obs[:, 9] = ball_vx
    obs[:, 10] = ball_vy
    obs[:, 11] = ball_vz
    obs[:, 12:15] = obs[:, 6:9] - obs[:, 0:3]
    obs[:, 15] = -3.5
    obs[:, 16] = 0.0
    obs[:, 17] = 0.6
    return obs


def _make_dataset(seed: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    policy = ScriptedPolicy(name="coach", side="p1")
    obs = _sample_obs(rng, samples)
    features = np.stack([rally_features(row) for row in obs], axis=0)
    actions = np.stack([policy.predict(row) for row in obs], axis=0)
    return features.astype(np.float32), actions.astype(np.float32)


def _trajectory_dataset(seed: int, rallies: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    model = mujoco.MjModel.from_xml_path(os.path.join("assets", "volleyball.xml"))
    data = mujoco.MjData(model)
    idx = _build_indices(model)
    policies = {"p1": ScriptedPolicy(name="coach-p1", side="p1"), "p2": ScriptedPolicy(name="coach-p2", side="p2")}
    feature_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []

    for rally in range(rallies):
        server = "p1" if rally % 2 == 0 else "p2"
        _serve(model, data, idx, server)
        last_touch = server
        touches = {"p1": 0, "p2": 0}
        cooldown = {"p1": 0, "p2": 0}
        prev_x = float(data.qpos[idx.ball_qpos])
        crossings = 0

        for _ in range(9000):
            obs = {"p1": _observation(data, idx, "p1"), "p2": _observation(data, idx, "p2")}
            policy_actions = {side: policies[side].predict(obs[side]) for side in ("p1", "p2")}
            for side in ("p1", "p2"):
                feature_rows.append(rally_features(obs[side]))
                action_rows.append(policy_actions[side])

            actual_actions = {side: _actual_action(policy_actions[side], side) for side in ("p1", "p2")}
            apply_action(data, actual_actions["p1"], idx.p1, enable_jump_impulse=True, enable_hit_up=True)
            apply_action(data, actual_actions["p2"], idx.p2, enable_jump_impulse=True, enable_hit_up=True)
            mujoco.mj_step(model, data)

            ball_x = float(data.qpos[idx.ball_qpos])
            ball_z = float(data.qpos[idx.ball_qpos + 2])
            if prev_x * ball_x < 0.0 and ball_z >= NET_HEIGHT:
                crossings += 1

            for side in cooldown:
                cooldown[side] = max(0, cooldown[side] - 1)

            for side, player in (("p1", idx.p1), ("p2", idx.p2)):
                touched = _contacted(data, idx, player.geom_id) or _reachable(data, idx, side, actual_actions[side])
                if cooldown[side] == 0 and touched:
                    if last_touch != side:
                        touches[side] = 0
                    touches[side] += 1
                    last_touch = side
                    cooldown[side] = 12
                    if touches[side] <= 3:
                        _apply_volleyball_touch(data, idx, side, touches[side])

            if crossings >= 5:
                break
            prev_x = ball_x

        # Add a little jitter around the most important states so the learned
        # model does not fall apart when physics produces tiny timing changes.
        if feature_rows:
            last_features = np.array(feature_rows[-200:], dtype=np.float32)
            last_actions = np.array(action_rows[-200:], dtype=np.float32)
            jitter = rng.normal(0.0, 0.025, size=last_features.shape).astype(np.float32)
            feature_rows.extend(last_features + jitter)
            action_rows.extend(last_actions)

    return np.array(feature_rows, dtype=np.float32), np.array(action_rows, dtype=np.float32)


def _train_model(features: np.ndarray, actions: np.ndarray, epochs: int, batch_size: int, lr: float):
    feature_mean = features.mean(axis=0).astype(np.float32)
    feature_scale = features.std(axis=0).astype(np.float32)
    feature_scale[feature_scale < 1e-6] = 1.0
    x = (features - feature_mean) / feature_scale

    y = actions.copy()
    y[:, 0:2] = np.arctanh(np.clip(y[:, 0:2], -0.95, 0.95))
    y[:, 2] = np.where(y[:, 2] > 0.5, 1.0, 0.0)

    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = RallyNet(input_size=x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    move_loss = nn.MSELoss()
    hit_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))

    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = move_loss(pred[:, 0:2], yb[:, 0:2]) + hit_loss(pred[:, 2:3], yb[:, 2:3])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(xb)
        print(f"epoch={epoch} loss={total / len(dataset):.4f}")

    model.eval()
    with torch.no_grad():
        raw = model(torch.from_numpy(x)).numpy()
    pred_actions = np.zeros_like(actions)
    pred_actions[:, 0] = np.tanh(raw[:, 0])
    pred_actions[:, 1] = np.tanh(raw[:, 1])
    pred_actions[:, 2] = 1.0 / (1.0 + np.exp(-raw[:, 2]))
    move_acc = np.mean(np.sign(pred_actions[:, 0:2]) == np.sign(actions[:, 0:2]))
    hit_acc = np.mean((pred_actions[:, 2] > 0.5) == (actions[:, 2] > 0.5))
    return model, feature_mean, feature_scale, move_acc, hit_acc


def _save_model(path: str, model: RallyNet, feature_mean: np.ndarray, feature_scale: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    layers = [module for module in model.net if isinstance(module, nn.Linear)]
    np.savez(
        path,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        w1=layers[0].weight.detach().numpy().T,
        b1=layers[0].bias.detach().numpy(),
        w2=layers[1].weight.detach().numpy().T,
        b2=layers[1].bias.detach().numpy(),
        w3=layers[2].weight.detach().numpy().T,
        b3=layers[2].bias.detach().numpy(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train lightweight rally policies by imitation from the rally coach.")
    parser.add_argument("--samples", type=int, default=120_000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random_features, random_actions = _make_dataset(seed=args.seed, samples=max(5_000, args.samples))
    trajectory_features, trajectory_actions = _trajectory_dataset(seed=args.seed + 1, rallies=24)
    features = np.concatenate([random_features, trajectory_features], axis=0)
    actions = np.concatenate([random_actions, trajectory_actions], axis=0)
    model, feature_mean, feature_scale, move_acc, hit_acc = _train_model(
        features,
        actions,
        epochs=max(1, args.epochs),
        batch_size=max(64, args.batch_size),
        lr=args.lr,
    )
    _save_model(os.path.join("models", "rally_p1.npz"), model, feature_mean, feature_scale)
    _save_model(os.path.join("models", "rally_p2.npz"), model, feature_mean, feature_scale)
    print(
        "saved rally models to models/rally_p1.npz and models/rally_p2.npz "
        f"move_sign_acc={move_acc:.3f} hit_acc={hit_acc:.3f}"
    )


if __name__ == "__main__":
    main()
