from __future__ import annotations

import argparse
import os
import random

import mujoco
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from play_2v2 import (
    CONTACT_COOLDOWN_STEPS,
    MODEL_PATH,
    NET_HEIGHT,
    TeamState,
    _apply_team_touch,
    _build_handles,
    _coach_actions_for_all,
    _contact_player,
    _reset_players,
    _serve,
    _team_for_player,
    player_features,
)
from volleyball_core import GROUND_Z_THRESHOLD, apply_action


class TeamNet(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 96),
            nn.Tanh(),
            nn.Linear(96, 96),
            nn.Tanh(),
            nn.Linear(96, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _collect_rollouts(seed: int, rallies: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    model = mujoco.MjModel.from_xml_path(os.path.join("assets", "volleyball.xml"))
    data = mujoco.MjData(model)
    h = _build_handles(model)
    features: list[np.ndarray] = []
    actions: list[np.ndarray] = []

    for rally in range(rallies):
        server = "blue" if rally % 2 == 0 else "orange"
        _serve(model, data, h, server)
        blue = TeamState(name="blue", side="+x", players=["blue_back", "blue_front"])
        orange = TeamState(name="orange", side="-x", players=["orange_back", "orange_front"])
        teams = {"blue": blue, "orange": orange}
        cooldown = {player_id: 0 for player_id in h.players}
        last_touch_team: str | None = None
        prev_x = float(data.qpos[h.ball_qpos])
        crossings = 0
        ball_entered_court = False

        for _ in range(7000):
            coach_actions = _coach_actions_for_all(data, h, blue, orange)
            for player_id in ("blue_back", "blue_front", "orange_back", "orange_front"):
                action = coach_actions[player_id].copy()
                if _team_for_player(player_id) == "orange":
                    action[1] *= -1.0
                features.append(player_features(data, h, player_id, blue, orange))
                actions.append(action)

            for player_id, action in coach_actions.items():
                apply_action(data, action, h.players[player_id], enable_jump_impulse=True, enable_hit_up=True)
            mujoco.mj_step(model, data)

            ball = data.qpos[h.ball_qpos:h.ball_qpos + 3].copy()
            if abs(float(ball[0])) <= 9.0 and abs(float(ball[1])) <= 4.5:
                ball_entered_court = True
            if prev_x * float(ball[0]) < 0.0 and ball[2] > NET_HEIGHT:
                crossings += 1
                blue.touches = 0
                orange.touches = 0

            for player_id in cooldown:
                cooldown[player_id] = max(0, cooldown[player_id] - 1)

            for player_id in ("blue_back", "blue_front", "orange_back", "orange_front"):
                if cooldown[player_id] > 0:
                    continue
                if not _contact_player(data, h, player_id, coach_actions[player_id][2] > 0.5):
                    continue
                team = _team_for_player(player_id)
                state = teams[team]
                if last_touch_team != team:
                    state.touches = 0
                state.touches += 1
                cooldown[player_id] = CONTACT_COOLDOWN_STEPS
                last_touch_team = team
                if state.touches <= 3:
                    _apply_team_touch(data, h, team, player_id, state.touches, rng)
                break

            if (ball_entered_court and (abs(float(ball[0])) > 9.0 or abs(float(ball[1])) > 4.5)) or ball[2] < GROUND_Z_THRESHOLD or crossings >= 3:
                break
            prev_x = float(ball[0])

        _reset_players(model, data, h)

    return np.asarray(features, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def _train(features: np.ndarray, actions: np.ndarray, epochs: int, batch_size: int, lr: float, seed: int):
    torch.manual_seed(seed)
    feature_mean = features.mean(axis=0).astype(np.float32)
    feature_scale = features.std(axis=0).astype(np.float32)
    feature_scale[feature_scale < 1e-6] = 1.0
    x = (features - feature_mean) / feature_scale

    y = actions.copy()
    y[:, 0:2] = np.arctanh(np.clip(y[:, 0:2], -0.95, 0.95))
    y[:, 2] = np.where(y[:, 2] > 0.5, 1.0, 0.0)

    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch_size, shuffle=True)
    net = TeamNet(x.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    move_loss = nn.MSELoss()
    hit_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([18.0]))

    for epoch in range(1, epochs + 1):
        total = 0.0
        for xb, yb in loader:
            pred = net(xb)
            loss = move_loss(pred[:, :2], yb[:, :2]) + hit_loss(pred[:, 2:3], yb[:, 2:3])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
        print(f"epoch={epoch} loss={total / len(x):.4f}")

    with torch.no_grad():
        raw = net(torch.from_numpy(x)).numpy()
    pred_hit = 1.0 / (1.0 + np.exp(-raw[:, 2]))
    hit_acc = np.mean((pred_hit > 0.5) == (actions[:, 2] > 0.5))
    move_acc = np.mean(np.sign(np.tanh(raw[:, :2])) == np.sign(actions[:, :2]))
    return net, feature_mean, feature_scale, float(move_acc), float(hit_acc)


def _save(path: str, net: TeamNet, feature_mean: np.ndarray, feature_scale: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    layers = [layer for layer in net.net if isinstance(layer, nn.Linear)]
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
    parser = argparse.ArgumentParser(description="Train the neural 2v2 volleyball team policy.")
    parser.add_argument("--rallies", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", default=MODEL_PATH)
    args = parser.parse_args()

    features, actions = _collect_rollouts(seed=args.seed, rallies=max(4, args.rallies))
    hit_mask = actions[:, 2] > 0.5
    if np.any(hit_mask):
        features = np.concatenate([features, np.repeat(features[hit_mask], 24, axis=0)], axis=0)
        actions = np.concatenate([actions, np.repeat(actions[hit_mask], 24, axis=0)], axis=0)
    print(f"collected samples={len(features)} hit_rate={np.mean(actions[:, 2] > 0.5):.3f}")
    net, feature_mean, feature_scale, move_acc, hit_acc = _train(
        features,
        actions,
        epochs=max(1, args.epochs),
        batch_size=max(64, args.batch_size),
        lr=args.lr,
        seed=args.seed,
    )
    _save(args.out, net, feature_mean, feature_scale)
    print(f"saved {args.out} move_sign_acc={move_acc:.3f} hit_acc={hit_acc:.3f}")


if __name__ == "__main__":
    main()
