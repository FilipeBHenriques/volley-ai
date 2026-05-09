from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from typing import Protocol

import mujoco
import mujoco.viewer
import numpy as np

from volleyball_core import GROUND_Z_THRESHOLD, PlayerHandles, _player_handles, apply_action, reset_ball
from volleyball_env import NET_HEIGHT

COURT_X_LIMIT = 9.0
COURT_Y_LIMIT = 4.5
CONTACT_RADIUS = 0.68
BLOCK_RADIUS = 0.85
MAX_RALLY_STEPS = 9000
CONTACT_COOLDOWN_STEPS = 24
MODEL_PATH = os.path.join("models", "team2v2_policy.npz")
SERVE_X = 9.75
HIT_THRESHOLD = 0.35


class TeamPolicy(Protocol):
    name: str

    def predict(self, features: np.ndarray) -> np.ndarray:
        ...


@dataclass
class TeamState:
    name: str
    side: str
    players: list[str]
    touches: int = 0
    last_touch: str | None = None


@dataclass
class Handles:
    players: dict[str, PlayerHandles]
    ball_body_id: int
    ball_geom_id: int
    ball_qpos: int
    ball_qvel: int


@dataclass
class RallyResult:
    winner: str
    reason: str
    crossings: int
    blue_touches: int
    orange_touches: int


class NeuralTeamPolicy:
    def __init__(self, path: str) -> None:
        self.name = "trained neural 2v2 policy"
        with np.load(path) as model:
            self.feature_mean = model["feature_mean"].astype(np.float32)
            self.feature_scale = model["feature_scale"].astype(np.float32)
            self.w1 = model["w1"].astype(np.float32)
            self.b1 = model["b1"].astype(np.float32)
            self.w2 = model["w2"].astype(np.float32)
            self.b2 = model["b2"].astype(np.float32)
            self.w3 = model["w3"].astype(np.float32)
            self.b3 = model["b3"].astype(np.float32)

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = (features.astype(np.float32) - self.feature_mean) / self.feature_scale
        h1 = np.tanh(x @ self.w1 + self.b1)
        h2 = np.tanh(h1 @ self.w2 + self.b2)
        raw = h2 @ self.w3 + self.b3
        return np.array([np.tanh(raw[0]), np.tanh(raw[1]), 1.0 / (1.0 + np.exp(-raw[2]))], dtype=np.float32)


class CoachTeamPolicy:
    name = "coach policy"

    def predict(self, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Coach actions are generated from full simulator state.")


def _build_handles(model: mujoco.MjModel) -> Handles:
    ball_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
    return Handles(
        players={
            "blue_back": _player_handles(model, ""),
            "orange_back": _player_handles(model, "2"),
            "blue_front": _player_handles(model, "3"),
            "orange_front": _player_handles(model, "4"),
        },
        ball_body_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball"),
        ball_geom_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom"),
        ball_qpos=model.jnt_qposadr[ball_joint],
        ball_qvel=model.jnt_dofadr[ball_joint],
    )


def _set_player_pose(model: mujoco.MjModel, data: mujoco.MjData, player: PlayerHandles, x: float, y: float, z: float = 1.0) -> None:
    body_pos = model.body_pos[player.body_id]
    data.qpos[player.slide_y_qpos] = y - float(body_pos[1])
    data.qpos[player.slide_x_qpos] = x - float(body_pos[0])
    data.qpos[player.slide_z_qpos] = z - float(body_pos[2])
    data.qvel[player.slide_y_qvel] = 0.0
    data.qvel[player.slide_x_qvel] = 0.0
    data.qvel[player.slide_z_qvel] = 0.0


def _player_pos(data: mujoco.MjData, player: PlayerHandles) -> np.ndarray:
    return data.xpos[player.body_id].copy()


def _ball_pos(data: mujoco.MjData, h: Handles) -> np.ndarray:
    return data.qpos[h.ball_qpos:h.ball_qpos + 3].copy()


def _ball_vel(data: mujoco.MjData, h: Handles) -> np.ndarray:
    return data.qvel[h.ball_qvel:h.ball_qvel + 3].copy()


def _reset_players(model: mujoco.MjModel, data: mujoco.MjData, h: Handles) -> None:
    _set_player_pose(model, data, h.players["blue_back"], 6.1, -1.7)
    _set_player_pose(model, data, h.players["blue_front"], 2.1, 1.7)
    _set_player_pose(model, data, h.players["orange_back"], -6.1, 1.7)
    _set_player_pose(model, data, h.players["orange_front"], -2.1, -1.7)


def _serve(model: mujoco.MjModel, data: mujoco.MjData, h: Handles, server: str) -> None:
    mujoco.mj_resetData(model, data)
    _reset_players(model, data, h)
    if server == "blue":
        pos = np.array([SERVE_X, -1.7, 2.25], dtype=np.float64)
        target = np.array([-6.1, 1.7, 1.65], dtype=np.float64)
    else:
        pos = np.array([-SERVE_X, 1.7, 2.25], dtype=np.float64)
        target = np.array([6.1, -1.7, 1.65], dtype=np.float64)
    vel = np.zeros(6, dtype=np.float64)
    reset_ball(model, data, h, pos=pos, vel=vel)
    _aim_ball(data, h, target, 1.9)


def _team_for_player(player_id: str) -> str:
    return "blue" if player_id.startswith("blue") else "orange"


def _sign(team: str) -> float:
    return -1.0 if team == "blue" else 1.0


def _other_team(team: str) -> str:
    return "orange" if team == "blue" else "blue"


def _teammate(player_id: str) -> str:
    return {
        "blue_back": "blue_front",
        "blue_front": "blue_back",
        "orange_back": "orange_front",
        "orange_front": "orange_back",
    }[player_id]


def _player_role(player_id: str) -> float:
    return 0.0 if player_id.endswith("back") else 1.0


def player_features(data: mujoco.MjData, h: Handles, player_id: str, blue: TeamState, orange: TeamState) -> np.ndarray:
    player = h.players[player_id]
    team = _team_for_player(player_id)
    state = blue if team == "blue" else orange
    pos = _player_pos(data, player)
    teammate_pos = _player_pos(data, h.players[_teammate(player_id)])
    ball = _ball_pos(data, h)
    ball_vel = _ball_vel(data, h)
    side_sign = 1.0 if team == "blue" else -1.0
    own_side = 1.0 if (ball[0] > 0.0 if team == "blue" else ball[0] < 0.0) else 0.0
    return np.array(
        [
            1.0,
            side_sign,
            _player_role(player_id),
            float(state.touches),
            own_side,
            pos[0] * side_sign,
            pos[1],
            pos[2],
            teammate_pos[0] * side_sign,
            teammate_pos[1],
            teammate_pos[2],
            ball[0] * side_sign,
            ball[1],
            ball[2],
            ball_vel[0] * side_sign,
            ball_vel[1],
            ball_vel[2],
            (ball[0] - pos[0]) * side_sign,
            ball[1] - pos[1],
            ball[2] - pos[2],
            float(np.linalg.norm(ball[:2] - pos[:2])),
            1.0 if abs(ball[0]) < 1.1 and ball[2] > NET_HEIGHT - 0.2 else 0.0,
        ],
        dtype=np.float32,
    )


def _move_toward(data: mujoco.MjData, player: PlayerHandles, target: np.ndarray, hit: bool = False) -> np.ndarray:
    pos = _player_pos(data, player)
    err_x = float(target[0] - pos[0])
    err_y = float(target[1] - pos[1])
    depth = 0.0 if abs(err_x) < 0.12 else float(np.sign(err_x))
    lat = 0.0 if abs(err_y) < 0.12 else float(np.sign(err_y))
    return np.array([lat, depth, 1.0 if hit else 0.0], dtype=np.float32)


def _team_targets(team: str, ball: np.ndarray, current_touches: int) -> dict[str, np.ndarray]:
    if team == "blue":
        home_back = np.array([6.1, -1.7, 1.0])
        home_front = np.array([2.0, 1.7, 1.0])
        defend_back = np.array([5.6, ball[1], 1.0])
        block_front = np.array([0.55, np.clip(ball[1], -3.0, 3.0), 1.0])
        set_front = np.array([1.55, 0.8, 1.0])
    else:
        home_back = np.array([-6.1, 1.7, 1.0])
        home_front = np.array([-2.0, -1.7, 1.0])
        defend_back = np.array([-5.6, ball[1], 1.0])
        block_front = np.array([-0.55, np.clip(ball[1], -3.0, 3.0), 1.0])
        set_front = np.array([-1.55, -0.8, 1.0])

    if current_touches == 0:
        return {"back": defend_back, "front": set_front}
    if current_touches == 1:
        return {"back": home_back, "front": set_front}
    return {"back": home_back, "front": block_front}


def _contact_player(data: mujoco.MjData, h: Handles, player_id: str, want_hit: bool) -> bool:
    pos = _player_pos(data, h.players[player_id])
    ball = _ball_pos(data, h)
    dist = float(np.linalg.norm(ball[:2] - pos[:2]))
    close_height = 0.65 < ball[2] < 3.45
    return bool(want_hit and dist < CONTACT_RADIUS and close_height)


def _can_play_ball(data: mujoco.MjData, h: Handles, player_id: str) -> bool:
    team = _team_for_player(player_id)
    pos = _player_pos(data, h.players[player_id])
    ball = _ball_pos(data, h)
    own_side = ball[0] > -0.25 if team == "blue" else ball[0] < 0.25
    dist = float(np.linalg.norm(ball[:2] - pos[:2]))
    return bool(own_side and dist < CONTACT_RADIUS and 0.65 < ball[2] < 3.45)


def _can_block(data: mujoco.MjData, h: Handles, player_id: str) -> bool:
    if player_id.endswith("back"):
        return False
    team = _team_for_player(player_id)
    pos = _player_pos(data, h.players[player_id])
    ball = _ball_pos(data, h)
    ball_vel = _ball_vel(data, h)
    moving_toward_team = ball_vel[0] > 0.0 if team == "blue" else ball_vel[0] < 0.0
    near_net = abs(ball[0]) < 1.05 and ball[2] > NET_HEIGHT - 0.2
    aligned = float(np.linalg.norm(ball[:2] - pos[:2])) < BLOCK_RADIUS
    return bool(near_net and moving_toward_team and aligned)


def _legalize_action(data: mujoco.MjData, h: Handles, player_id: str, action: np.ndarray) -> np.ndarray:
    clean = np.clip(action.astype(np.float32), -1.0, 1.0)
    wants_contact = clean[2] > HIT_THRESHOLD
    clean[2] = 1.0 if wants_contact and (_can_play_ball(data, h, player_id) or _can_block(data, h, player_id)) else 0.0
    return clean


def _aim_ball(data: mujoco.MjData, h: Handles, target: np.ndarray, flight_time: float) -> None:
    ball = _ball_pos(data, h)
    vel = data.qvel[h.ball_qvel:h.ball_qvel + 6]
    vel[:3] = [
        (float(target[0]) - float(ball[0])) / flight_time,
        (float(target[1]) - float(ball[1])) / flight_time,
        (float(target[2]) - float(ball[2]) + 0.5 * 9.81 * flight_time * flight_time) / flight_time,
    ]
    vel[3:6] = 0.0


def _apply_team_touch(data: mujoco.MjData, h: Handles, team: str, player_id: str, touch_number: int, rng: random.Random) -> str:
    if touch_number == 1:
        target = np.array([2.0, 1.7, 2.35]) if team == "blue" else np.array([-2.0, -1.7, 2.35])
        _aim_ball(data, h, target, 1.05)
        return "receive"

    if touch_number == 2:
        target_x = rng.uniform(-7.2, -4.0) if team == "blue" else rng.uniform(4.0, 7.2)
        target_y = rng.choice([-1.0, 1.0]) * rng.uniform(0.9, 3.0)
        target = np.array([target_x, target_y, rng.uniform(0.85, 1.35)])
        _aim_ball(data, h, target, rng.uniform(0.78, 0.98))
        return "spike"

    target_x = rng.uniform(-7.4, -3.8) if team == "blue" else rng.uniform(3.8, 7.4)
    target_y = rng.choice([-1.0, 1.0]) * rng.uniform(1.0, 3.3)
    target = np.array([target_x, target_y, rng.uniform(0.75, 1.35)])
    _aim_ball(data, h, target, rng.uniform(0.70, 0.92))
    data.qvel[h.ball_qvel + 2] += rng.uniform(-0.8, 0.35)
    return "spike"


def _try_block(data: mujoco.MjData, h: Handles, attacking_team: str, rng: random.Random) -> str | None:
    defending_team = _other_team(attacking_team)
    blocker_id = "blue_front" if defending_team == "blue" else "orange_front"
    blocker = h.players[blocker_id]
    blocker_pos = _player_pos(data, blocker)
    ball = _ball_pos(data, h)
    ball_vel = _ball_vel(data, h)
    near_net = abs(ball[0]) < 0.9 and ball[2] > NET_HEIGHT - 0.15
    moving_over = ball_vel[0] > 0.0 if attacking_team == "orange" else ball_vel[0] < 0.0
    if not near_net or not moving_over:
        return None
    if float(np.linalg.norm(ball[:2] - blocker_pos[:2])) > BLOCK_RADIUS:
        return None
    if rng.random() > 0.42:
        return None

    target_x = rng.uniform(2.8, 5.6) if defending_team == "orange" else rng.uniform(-5.6, -2.8)
    target_y = rng.uniform(-2.8, 2.8)
    _aim_ball(data, h, np.array([target_x, target_y, 0.9]), 0.55)
    return defending_team


def _coach_actions_for_all(data: mujoco.MjData, h: Handles, blue: TeamState, orange: TeamState) -> dict[str, np.ndarray]:
    ball = _ball_pos(data, h)
    on_blue_side = ball[0] > 0.0
    blue_targets = _team_targets("blue", ball, blue.touches if on_blue_side else 2)
    orange_targets = _team_targets("orange", ball, orange.touches if not on_blue_side else 2)

    actions: dict[str, np.ndarray] = {}
    for player_id, role in (("blue_back", "back"), ("blue_front", "front")):
        target = blue_targets[role]
        want_hit = _can_play_ball(data, h, player_id) or _can_block(data, h, player_id)
        actions[player_id] = _move_toward(data, h.players[player_id], target, hit=want_hit)

    for player_id, role in (("orange_back", "back"), ("orange_front", "front")):
        target = orange_targets[role]
        want_hit = _can_play_ball(data, h, player_id) or _can_block(data, h, player_id)
        actions[player_id] = _move_toward(data, h.players[player_id], target, hit=want_hit)
    return actions


def _model_actions_for_all(data: mujoco.MjData, h: Handles, blue: TeamState, orange: TeamState, policy: TeamPolicy) -> dict[str, np.ndarray]:
    actions = {}
    for player_id in ("blue_back", "blue_front", "orange_back", "orange_front"):
        action = policy.predict(player_features(data, h, player_id, blue, orange))
        if _team_for_player(player_id) == "orange":
            action = action.copy()
            action[1] *= -1.0
        actions[player_id] = _legalize_action(data, h, player_id, action)
    return actions


def _ground_winner(ball_x: float) -> str:
    return "orange" if ball_x > 0.0 else "blue"


def _out_winner(last_touch_team: str | None, ball_x: float) -> str:
    if last_touch_team is not None:
        return _other_team(last_touch_team)
    return "orange" if ball_x > 0.0 else "blue"


def _run_rally(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    h: Handles,
    server: str,
    *,
    render: bool,
    viewer,
    rng: random.Random,
    policy: TeamPolicy | None,
) -> RallyResult:
    _serve(model, data, h, server)
    blue = TeamState(name="blue", side="+x", players=["blue_back", "blue_front"])
    orange = TeamState(name="orange", side="-x", players=["orange_back", "orange_front"])
    teams = {"blue": blue, "orange": orange}
    last_touch_team: str | None = None
    prev_x = float(data.qpos[h.ball_qpos])
    crossings = 0
    touch_totals = {"blue": 0, "orange": 0}
    last_attack_team: str | None = None
    cooldown = {player_id: 0 for player_id in h.players}
    ball_entered_court = False

    for step in range(1, MAX_RALLY_STEPS + 1):
        actions = (
            _model_actions_for_all(data, h, blue, orange, policy)
            if policy is not None
            else _coach_actions_for_all(data, h, blue, orange)
        )
        for player_id, action in actions.items():
            apply_action(data, action, h.players[player_id], enable_jump_impulse=True, enable_hit_up=True)

        mujoco.mj_step(model, data)
        ball = _ball_pos(data, h)
        if abs(ball[0]) <= COURT_X_LIMIT and abs(ball[1]) <= COURT_Y_LIMIT:
            ball_entered_court = True
        for player_id in cooldown:
            cooldown[player_id] = max(0, cooldown[player_id] - 1)

        if prev_x * ball[0] < 0.0 and ball[2] > NET_HEIGHT:
            crossings += 1
            blue.touches = 0
            orange.touches = 0

        if last_attack_team is not None:
            blocked_by = _try_block(data, h, last_attack_team, rng)
            if blocked_by is not None:
                last_touch_team = blocked_by
                last_attack_team = None

        for player_id in ("blue_back", "blue_front", "orange_back", "orange_front"):
            if cooldown[player_id] > 0:
                continue
            team = _team_for_player(player_id)
            if not _contact_player(data, h, player_id, actions[player_id][2] > 0.5):
                continue

            state = teams[team]
            if last_touch_team != team:
                state.touches = 0
            state.touches += 1
            touch_totals[team] += 1
            cooldown[player_id] = CONTACT_COOLDOWN_STEPS
            last_touch_team = team
            state.last_touch = player_id
            if state.touches > 3:
                return RallyResult(_other_team(team), f"{team} used more than 3 touches", crossings, touch_totals["blue"], touch_totals["orange"])

            kind = _apply_team_touch(data, h, team, player_id, state.touches, rng)
            last_attack_team = team if kind == "spike" else None
            break

        if ball_entered_court and (abs(ball[0]) > COURT_X_LIMIT or abs(ball[1]) > COURT_Y_LIMIT):
            return RallyResult(_out_winner(last_touch_team, float(ball[0])), "ball out", crossings, touch_totals["blue"], touch_totals["orange"])

        if ball[2] < GROUND_Z_THRESHOLD:
            return RallyResult(_ground_winner(float(ball[0])), "ball grounded", crossings, touch_totals["blue"], touch_totals["orange"])

        prev_x = float(ball[0])

        if render and viewer is not None and step % 5 == 0:
            viewer.sync()
            time.sleep(model.opt.timestep * 5)

    return RallyResult(_other_team(server), "rally step limit", crossings, touch_totals["blue"], touch_totals["orange"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a competitive trained 2v2 volleyball match.")
    parser.add_argument("--points", type=int, default=15)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--coach", action="store_true", help="Debug with the hand-authored coach instead of trained weights.")
    parser.add_argument("--model-path", default=MODEL_PATH)
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    model = mujoco.MjModel.from_xml_path(os.path.join(here, "assets", "volleyball.xml"))
    data = mujoco.MjData(model)
    h = _build_handles(model)
    rng = random.Random(args.seed)
    policy: TeamPolicy | None = None
    if not args.coach:
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Cannot find trained 2v2 model at {args.model_path}. Run train_2v2_models.py first.")
        policy = NeuralTeamPolicy(args.model_path)
    score = {"blue": 0, "orange": 0}
    server = "blue"

    policy_name = "coach/debug" if policy is None else policy.name
    print(f"2v2 match start: Blue vs Orange using {policy_name}. First to {args.points}.")
    viewer_ctx = mujoco.viewer.launch_passive(model, data) if args.render else None
    try:
        viewer = viewer_ctx.__enter__() if viewer_ctx is not None else None
        if viewer is not None:
            viewer.cam.lookat[:] = [0.0, 0.0, 1.5]
            viewer.cam.distance = 17.5
            viewer.cam.elevation = -17.0
            viewer.cam.azimuth = 90.0

        point = 0
        while max(score.values()) < args.points:
            point += 1
            result = _run_rally(model, data, h, server, render=args.render, viewer=viewer, rng=rng, policy=policy)
            score[result.winner] += 1
            server = result.winner
            print(
                f"point={point} winner={result.winner.upper()} score={score['blue']}-{score['orange']} "
                f"reason={result.reason} crossings={result.crossings} touches=blue:{result.blue_touches}/orange:{result.orange_touches}"
            )

        winner = "blue" if score["blue"] > score["orange"] else "orange"
        print(f"Winner: {winner.upper()} final_score={score['blue']}-{score['orange']}")
    finally:
        if viewer_ctx is not None:
            viewer_ctx.__exit__(None, None, None)


if __name__ == "__main__":
    main()
