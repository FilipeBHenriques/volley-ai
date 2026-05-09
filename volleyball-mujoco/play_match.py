from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Protocol

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from volleyball_core import GROUND_Z_THRESHOLD, ModelIndices, _build_indices, apply_action, reset_ball
from volleyball_env import NET_HEIGHT, OBS_CLIP_POS, OBS_CLIP_VEL, VolleyballEnv

COURT_X_LIMIT = 9.0
COURT_Y_LIMIT = 4.5
DEFAULT_POINT_LIMIT = 15
CONTACT_COOLDOWN_STEPS = 12
MAX_RALLY_STEPS = 12000
TARGET_RALLY_CROSSES = 4
REACH_RADIUS = 0.8
REACH_MIN_Z = 0.55
REACH_MAX_Z = 3.25
THIRD_TOUCH_RETURN_WINDOW = 300


class Policy(Protocol):
    name: str

    def predict(self, obs: np.ndarray) -> np.ndarray:
        ...


@dataclass
class RallyResult:
    winner: str
    reason: str
    steps: int
    touches_p1: int
    touches_p2: int
    crossings: int


def rally_features(obs: np.ndarray) -> np.ndarray:
    player_pos = obs[0:3]
    ball_pos = obs[6:9]
    ball_vel = obs[9:12]
    rel = obs[12:15]
    horiz_dist = float(np.linalg.norm(rel[:2]))
    same_side = 1.0 if ball_pos[0] > 0.0 else 0.0
    close = 1.0 if horiz_dist < REACH_RADIUS else 0.0
    height_window = 1.0 if 0.7 < ball_pos[2] < 3.1 else 0.0
    descending = 1.0 if ball_vel[2] < 1.5 else 0.0
    return np.array(
        [
            1.0,
            player_pos[0],
            player_pos[1],
            player_pos[2],
            ball_pos[0],
            ball_pos[1],
            ball_pos[2],
            ball_vel[0],
            ball_vel[1],
            ball_vel[2],
            rel[0],
            rel[1],
            rel[2],
            horiz_dist,
            same_side,
            close,
            height_window,
            descending,
            close * height_window * descending,
        ],
        dtype=np.float32,
    )


class ScriptedPolicy:
    def __init__(self, name: str, side: str) -> None:
        self.name = name
        self.side = side

    def predict(self, obs: np.ndarray) -> np.ndarray:
        player_pos = obs[0:3]
        ball_pos = obs[6:9]
        ball_vel = obs[9:12]

        target_x = float(np.clip(ball_pos[0], 0.45, 8.0))
        target_y = float(np.clip(ball_pos[1], -3.8, 3.8))
        if ball_pos[0] < 0.0:
            target_x = 3.5
            target_y = 0.0

        err_x = target_x - float(player_pos[0])
        err_y = target_y - float(player_pos[1])
        depth = 0.0 if abs(err_x) < 0.15 else float(np.sign(err_x))
        lat = 0.0 if abs(err_y) < 0.15 else float(np.sign(err_y))

        horiz_dist = float(np.linalg.norm(ball_pos[:2] - player_pos[:2]))
        hit = 1.0 if horiz_dist < REACH_RADIUS and 0.7 < ball_pos[2] < 3.1 and ball_vel[2] < 1.5 else 0.0
        return np.array([lat, depth, hit], dtype=np.float32)


class RallyModelPolicy:
    def __init__(self, name: str, model_path: str) -> None:
        self.name = name
        with np.load(model_path) as model:
            self._feature_mean = model["feature_mean"].astype(np.float32)
            self._feature_scale = model["feature_scale"].astype(np.float32)
            self._weights = model["weights"].astype(np.float32) if "weights" in model else None
            self._w1 = model["w1"].astype(np.float32) if "w1" in model else None
            self._b1 = model["b1"].astype(np.float32) if "b1" in model else None
            self._w2 = model["w2"].astype(np.float32) if "w2" in model else None
            self._b2 = model["b2"].astype(np.float32) if "b2" in model else None
            self._w3 = model["w3"].astype(np.float32) if "w3" in model else None
            self._b3 = model["b3"].astype(np.float32) if "b3" in model else None

    def predict(self, obs: np.ndarray) -> np.ndarray:
        features = rally_features(obs)
        normalized = (features - self._feature_mean) / self._feature_scale
        if self._weights is not None:
            raw = normalized @ self._weights
        else:
            h1 = np.tanh(normalized @ self._w1 + self._b1)
            h2 = np.tanh(h1 @ self._w2 + self._b2)
            raw = h2 @ self._w3 + self._b3
        action = np.array([np.tanh(raw[0]), np.tanh(raw[1]), 1.0 / (1.0 + np.exp(-raw[2]))], dtype=np.float32)
        return np.clip(action, -1.0, 1.0)


class PPOPolicy:
    def __init__(self, name: str, model_path: str, vecnorm_path: str | None, side: str) -> None:
        self.name = name
        self.side = side
        self._vecnorm = None
        self._model = PPO.load(model_path)

        if vecnorm_path and os.path.exists(vecnorm_path):
            env = DummyVecEnv([lambda: VolleyballEnv(render_mode=None)])
            self._vecnorm = VecNormalize.load(vecnorm_path, env)
            self._vecnorm.training = False
            self._vecnorm.norm_reward = False

    def predict(self, obs: np.ndarray) -> np.ndarray:
        model_obs = obs.reshape(1, -1)
        if self._vecnorm is not None:
            model_obs = self._vecnorm.normalize_obs(model_obs)
        action, _ = self._model.predict(model_obs, deterministic=True)
        return np.asarray(action[0], dtype=np.float32)


def _set_player_pose(model: mujoco.MjModel, data: mujoco.MjData, player, x: float, y: float, z: float) -> None:
    body_pos = model.body_pos[player.body_id]
    data.qpos[player.slide_y_qpos] = y - float(body_pos[1])
    data.qpos[player.slide_x_qpos] = x - float(body_pos[0])
    data.qpos[player.slide_z_qpos] = z - float(body_pos[2])
    data.qvel[player.slide_y_qvel] = 0.0
    data.qvel[player.slide_x_qvel] = 0.0
    data.qvel[player.slide_z_qvel] = 0.0


def _player_velocity(data: mujoco.MjData, player) -> np.ndarray:
    return np.array(
        [
            data.qvel[player.slide_x_qvel],
            data.qvel[player.slide_y_qvel],
            data.qvel[player.slide_z_qvel],
        ],
        dtype=np.float64,
    )


def _observation(data: mujoco.MjData, idx: ModelIndices, side: str) -> np.ndarray:
    player = idx.p1 if side == "p1" else idx.p2
    opponent = idx.p2 if side == "p1" else idx.p1

    player_pos = data.xpos[player.body_id].copy()
    opponent_pos = data.xpos[opponent.body_id].copy()
    player_vel = _player_velocity(data, player)
    ball_pos = data.qpos[idx.ball_qpos:idx.ball_qpos + 3].copy()
    ball_vel = data.qvel[idx.ball_qvel:idx.ball_qvel + 3].copy()

    if side == "p2":
        player_pos[0] *= -1.0
        opponent_pos[0] *= -1.0
        player_vel[0] *= -1.0
        ball_pos[0] *= -1.0
        ball_vel[0] *= -1.0

    rel = ball_pos - player_pos
    obs = np.array(
        [
            player_pos[0],
            player_pos[1],
            player_pos[2],
            player_vel[0],
            player_vel[1],
            player_vel[2],
            ball_pos[0],
            ball_pos[1],
            ball_pos[2],
            ball_vel[0],
            ball_vel[1],
            ball_vel[2],
            rel[0],
            rel[1],
            rel[2],
            opponent_pos[0],
            opponent_pos[1],
            opponent_pos[2],
        ],
        dtype=np.float32,
    )

    pos_idx = np.array([0, 1, 2, 6, 7, 8, 12, 13, 14, 15, 16, 17])
    vel_idx = np.array([3, 4, 5, 9, 10, 11])
    obs[pos_idx] = np.clip(obs[pos_idx], -OBS_CLIP_POS, OBS_CLIP_POS)
    obs[vel_idx] = np.clip(obs[vel_idx], -OBS_CLIP_VEL, OBS_CLIP_VEL)
    return obs


def _actual_action(policy_action: np.ndarray, side: str) -> np.ndarray:
    action = np.clip(np.asarray(policy_action, dtype=np.float32), -1.0, 1.0)
    if side == "p2":
        action = action.copy()
        action[1] *= -1.0
    return action


def _contacted(data: mujoco.MjData, idx: ModelIndices, player_geom_id: int) -> bool:
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 == idx.ball_geom_id and geom2 == player_geom_id:
            return True
        if geom2 == idx.ball_geom_id and geom1 == player_geom_id:
            return True
    return False


def _reachable(data: mujoco.MjData, idx: ModelIndices, side: str, action: np.ndarray) -> bool:
    player = idx.p1 if side == "p1" else idx.p2
    player_pos = data.xpos[player.body_id]
    ball_pos = data.qpos[idx.ball_qpos:idx.ball_qpos + 3]
    horiz_dist = float(np.linalg.norm(ball_pos[:2] - player_pos[:2]))
    on_playable_side = ball_pos[0] > -0.25 if side == "p1" else ball_pos[0] < 0.25
    return bool(on_playable_side and action[2] > 0.5 and horiz_dist < REACH_RADIUS and REACH_MIN_Z < ball_pos[2] < REACH_MAX_Z)


def _apply_volleyball_touch(data: mujoco.MjData, idx: ModelIndices, side: str, touch_number: int) -> None:
    direction = -1.0 if side == "p1" else 1.0
    ball_pos = data.qpos[idx.ball_qpos:idx.ball_qpos + 3]
    ball_vel = data.qvel[idx.ball_qvel:idx.ball_qvel + 6]

    if touch_number == 1:
        ball_vel[:3] = [0.7 * direction, 0.0, 4.6]
    elif touch_number == 2:
        ball_vel[:3] = [0.9 * direction, 0.0, 5.2]
    else:
        target_x = -3.5 if side == "p1" else 3.5
        target_y = 0.0
        target_z = 1.9
        flight_time = 1.05
        ball_vel[:3] = [
            (target_x - float(ball_pos[0])) / flight_time,
            (target_y - float(ball_pos[1])) / flight_time,
            (target_z - float(ball_pos[2]) + 0.5 * 9.81 * flight_time * flight_time) / flight_time,
        ]
    ball_vel[3:6] = 0.0


def _point_winner_for_ground(ball_x: float) -> tuple[str, str]:
    if ball_x > 0.0:
        return "p2", "ball landed on P1 side"
    return "p1", "ball landed on P2 side"


def _serve(model: mujoco.MjModel, data: mujoco.MjData, idx: ModelIndices, server: str) -> None:
    mujoco.mj_resetData(model, data)
    _set_player_pose(model, data, idx.p1, x=3.5, y=0.0, z=1.0)
    _set_player_pose(model, data, idx.p2, x=-3.5, y=0.0, z=1.0)

    if server == "p1":
        pos = np.array([-3.5, 0.0, 3.0], dtype=np.float64)
        vel = np.array([0.0, 0.0, -0.2, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        pos = np.array([3.5, 0.0, 3.0], dtype=np.float64)
        vel = np.array([0.0, 0.0, -0.2, 0.0, 0.0, 0.0], dtype=np.float64)

    reset_ball(model, data, idx, pos=pos, vel=vel)


def _run_rally(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    idx: ModelIndices,
    p1_policy: Policy,
    p2_policy: Policy,
    server: str,
    *,
    frame_skip: int,
    render: bool,
    viewer,
) -> RallyResult:
    _serve(model, data, idx, server)
    last_touch_side: str | None = server
    touches = {"p1": 0, "p2": 0}
    cooldown = {"p1": 0, "p2": 0}
    previous_ball_x = float(data.qpos[idx.ball_qpos])
    pending_return_side: str | None = None
    pending_return_steps = 0
    crossings = 0

    for step in range(1, MAX_RALLY_STEPS + 1):
        p1_action = _actual_action(p1_policy.predict(_observation(data, idx, "p1")), "p1")
        p2_action = _actual_action(p2_policy.predict(_observation(data, idx, "p2")), "p2")
        actions = {"p1": p1_action, "p2": p2_action}
        apply_action(data, p1_action, idx.p1, enable_jump_impulse=True, enable_hit_up=True)
        apply_action(data, p2_action, idx.p2, enable_jump_impulse=True, enable_hit_up=True)

        for _ in range(frame_skip):
            mujoco.mj_step(model, data)
            ball_x = float(data.qpos[idx.ball_qpos])
            ball_y = float(data.qpos[idx.ball_qpos + 1])
            ball_z = float(data.qpos[idx.ball_qpos + 2])

            for side in cooldown:
                cooldown[side] = max(0, cooldown[side] - 1)

            if previous_ball_x * ball_x < 0.0 and ball_z < NET_HEIGHT:
                loser = last_touch_side or ("p1" if previous_ball_x > 0.0 else "p2")
                winner = "p2" if loser == "p1" else "p1"
                return RallyResult(winner, "ball crossed below net height", step, touches["p1"], touches["p2"], crossings)
            if previous_ball_x * ball_x < 0.0 and ball_z >= NET_HEIGHT:
                crossings += 1
                pending_return_side = None
                pending_return_steps = 0

            for side, player in (("p1", idx.p1), ("p2", idx.p2)):
                touched = _contacted(data, idx, player.geom_id) or _reachable(data, idx, side, actions[side])
                if cooldown[side] == 0 and touched:
                    if last_touch_side != side:
                        touches[side] = 0
                    touches[side] += 1
                    last_touch_side = side
                    cooldown[side] = CONTACT_COOLDOWN_STEPS
                    if touches[side] > 3:
                        winner = "p2" if side == "p1" else "p1"
                        return RallyResult(winner, f"{side.upper()} used more than 3 touches", step, touches["p1"], touches["p2"], crossings)
                    _apply_volleyball_touch(data, idx, side, touches[side])
                    if crossings >= TARGET_RALLY_CROSSES and touches[side] == 3:
                        winner = "p2" if server == "p1" else "p1"
                        return RallyResult(
                            winner,
                            f"long rally completed with {crossings} clean net crossings",
                            step,
                            touches["p1"],
                            touches["p2"],
                            crossings,
                        )
                    if touches[side] == 3:
                        pending_return_side = side
                        pending_return_steps = 0

            if pending_return_side is not None:
                pending_return_steps += 1
                own_side = ball_x > 0.0 if pending_return_side == "p1" else ball_x < 0.0
                if own_side and pending_return_steps > THIRD_TOUCH_RETURN_WINDOW:
                    winner = "p2" if pending_return_side == "p1" else "p1"
                    return RallyResult(
                        winner,
                        f"{pending_return_side.upper()} failed to return after 3 touches",
                        step,
                        touches["p1"],
                        touches["p2"],
                        crossings,
                    )

            if abs(ball_x) > COURT_X_LIMIT or abs(ball_y) > COURT_Y_LIMIT:
                loser = last_touch_side or ("p1" if ball_x > 0.0 else "p2")
                winner = "p2" if loser == "p1" else "p1"
                return RallyResult(winner, "ball went out after last touch", step, touches["p1"], touches["p2"], crossings)

            if ball_z < GROUND_Z_THRESHOLD:
                winner, reason = _point_winner_for_ground(ball_x)
                return RallyResult(winner, reason, step, touches["p1"], touches["p2"], crossings)

            previous_ball_x = ball_x

        if render and viewer is not None:
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep * frame_skip))

    winner = "p2" if server == "p1" else "p1"
    return RallyResult(winner, "rally step limit reached", MAX_RALLY_STEPS, touches["p1"], touches["p2"], crossings)


def _default_model_path(stage: int) -> str:
    return os.path.join("models", f"ppo_stage{stage}.zip")


def _default_vecnorm_path(stage: int) -> str:
    return os.path.join("models", f"vecnormalize_stage{stage}.pkl")


def _default_rally_model_path(side: str) -> str:
    return os.path.join("models", f"rally_{side}.npz")


def _build_policy(kind: str, side: str, model_path: str | None, vecnorm_path: str | None, stage: int) -> Policy:
    if kind == "scripted":
        return ScriptedPolicy(name=f"{side.upper()} scripted", side=side)

    if kind == "rally":
        resolved_model = model_path or _default_rally_model_path(side)
        if not os.path.exists(resolved_model):
            raise FileNotFoundError(f"Cannot find {side.upper()} rally model at {resolved_model}. Run train_rally_models.py first.")
        return RallyModelPolicy(name=f"{side.upper()} trained rally", model_path=resolved_model)

    resolved_model = model_path or _default_model_path(stage)
    resolved_vecnorm = vecnorm_path or _default_vecnorm_path(stage)
    if not os.path.exists(resolved_model):
        raise FileNotFoundError(f"Cannot find {side.upper()} model at {resolved_model}")
    return PPOPolicy(name=f"{side.upper()} PPO", model_path=resolved_model, vecnorm_path=resolved_vecnorm, side=side)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a 1v1 volleyball match to 15 between two policies.")
    parser.add_argument("--p1", choices=["ppo", "rally", "scripted"], default="rally", help="Policy type for Player 1.")
    parser.add_argument("--p2", choices=["ppo", "rally", "scripted"], default="rally", help="Policy type for Player 2.")
    parser.add_argument("--p1-model", default=None, help="Path to Player 1 PPO .zip.")
    parser.add_argument("--p2-model", default=None, help="Path to Player 2 PPO .zip.")
    parser.add_argument("--p1-vecnorm", default=None, help="Path to Player 1 VecNormalize .pkl.")
    parser.add_argument("--p2-vecnorm", default=None, help="Path to Player 2 VecNormalize .pkl.")
    parser.add_argument("--stage", type=int, default=3, help="Default model stage when model paths are omitted.")
    parser.add_argument("--points", type=int, default=DEFAULT_POINT_LIMIT, help="Points needed to win.")
    parser.add_argument("--win-by", type=int, default=1, help="Win-by margin. Use 2 for traditional deuce.")
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--render", action="store_true", help="Show the MuJoCo viewer while the match runs.")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    model = mujoco.MjModel.from_xml_path(os.path.join(here, "assets", "volleyball.xml"))
    data = mujoco.MjData(model)
    idx = _build_indices(model)

    p1_policy = _build_policy(args.p1, "p1", args.p1_model, args.p1_vecnorm, args.stage)
    p2_policy = _build_policy(args.p2, "p2", args.p2_model, args.p2_vecnorm, args.stage)
    score = {"p1": 0, "p2": 0}
    server = "p1"

    print(f"Match start: {p1_policy.name} vs {p2_policy.name}. First to {args.points}.")

    viewer_ctx = mujoco.viewer.launch_passive(model, data) if args.render else None
    try:
        viewer = viewer_ctx.__enter__() if viewer_ctx is not None else None
        if viewer is not None:
            viewer.cam.lookat[:] = [0.0, 0.0, 1.4]
            viewer.cam.distance = 17.0
            viewer.cam.elevation = -15.0
            viewer.cam.azimuth = 90.0

        point = 0
        while True:
            point += 1
            result = _run_rally(
                model,
                data,
                idx,
                p1_policy,
                p2_policy,
                server,
                frame_skip=max(1, args.frame_skip),
                render=args.render,
                viewer=viewer,
            )
            score[result.winner] += 1
            server = result.winner
            print(
                f"point={point} winner={result.winner.upper()} score={score['p1']}-{score['p2']} "
                f"reason={result.reason} crossings={result.crossings} touches=P1:{result.touches_p1}/P2:{result.touches_p2}"
            )

            leader = "p1" if score["p1"] >= score["p2"] else "p2"
            trailer = "p2" if leader == "p1" else "p1"
            if score[leader] >= args.points and score[leader] - score[trailer] >= args.win_by:
                print(f"Winner: {leader.upper()} final_score={score['p1']}-{score['p2']}")
                break
    finally:
        if viewer_ctx is not None:
            viewer_ctx.__exit__(None, None, None)


if __name__ == "__main__":
    main()
