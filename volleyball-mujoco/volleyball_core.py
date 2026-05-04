"""
Shared MuJoCo constants, handles, and simulator helpers.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import mujoco
import numpy as np

MOVE_FORCE: float = 450.0
JUMP_HEIGHT: float = 1.0
JUMP_VEL: float = math.sqrt(2.0 * 9.81 * JUMP_HEIGHT)
HIT_UP_FORCE: float = 1800.0
GROUND_TOL: float = 0.05
GROUND_Z_THRESHOLD: float = 0.12
BALL_INIT_POS: np.ndarray = np.array([3.5, 0.0, 4.0], dtype=np.float64)
BALL_INIT_VEL: np.ndarray = np.zeros(6, dtype=np.float64)


@dataclass
class PlayerHandles:
    body_id: int
    geom_id: int
    act_y_id: int
    act_x_id: int
    act_z_id: int
    slide_y_qpos: int
    slide_x_qpos: int
    slide_z_qpos: int
    slide_y_qvel: int
    slide_x_qvel: int
    slide_z_qvel: int


@dataclass
class ModelIndices:
    p1: PlayerHandles
    p2: PlayerHandles
    ball_body_id: int
    ball_geom_id: int
    ball_qpos: int
    ball_qvel: int


@dataclass
class SimState:
    misses: int = 0
    miss_cooldown_until: float = 0.0


def _player_handles(model: mujoco.MjModel, suffix: str) -> PlayerHandles:
    n2i = mujoco.mj_name2id
    body_id = n2i(model, mujoco.mjtObj.mjOBJ_BODY, f"player{suffix}")
    geom_id = n2i(model, mujoco.mjtObj.mjOBJ_GEOM, f"player{suffix}_geom" if suffix else "player_geom")
    act_y_id = n2i(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_y{suffix}")
    act_x_id = n2i(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_x{suffix}")
    act_z_id = n2i(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_z{suffix}")
    slide_y = n2i(model, mujoco.mjtObj.mjOBJ_JOINT, f"slide_y{suffix}")
    slide_x = n2i(model, mujoco.mjtObj.mjOBJ_JOINT, f"slide_x{suffix}")
    slide_z = n2i(model, mujoco.mjtObj.mjOBJ_JOINT, f"slide_z{suffix}")
    return PlayerHandles(
        body_id=body_id,
        geom_id=geom_id,
        act_y_id=act_y_id,
        act_x_id=act_x_id,
        act_z_id=act_z_id,
        slide_y_qpos=model.jnt_qposadr[slide_y],
        slide_x_qpos=model.jnt_qposadr[slide_x],
        slide_z_qpos=model.jnt_qposadr[slide_z],
        slide_y_qvel=model.jnt_dofadr[slide_y],
        slide_x_qvel=model.jnt_dofadr[slide_x],
        slide_z_qvel=model.jnt_dofadr[slide_z],
    )


def _build_indices(model: mujoco.MjModel) -> ModelIndices:
    ball_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
    return ModelIndices(
        p1=_player_handles(model, ""),
        p2=_player_handles(model, "2"),
        ball_body_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball"),
        ball_geom_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom"),
        ball_qpos=model.jnt_qposadr[ball_joint],
        ball_qvel=model.jnt_dofadr[ball_joint],
    )


def grounded(data: mujoco.MjData, p: PlayerHandles) -> bool:
    return float(data.qpos[p.slide_z_qpos]) < GROUND_TOL


def apply_action(
    data: mujoco.MjData,
    action: np.ndarray,
    p: PlayerHandles,
    *,
    enable_jump_impulse: bool = True,
    enable_hit_up: bool = True,
) -> None:
    lat = float(np.clip(action[0], -1.0, 1.0))
    depth = float(np.clip(action[1], -1.0, 1.0))
    hit_up = float(action[2])
    is_grounded = grounded(data, p)

    data.ctrl[p.act_y_id] = lat * MOVE_FORCE
    data.ctrl[p.act_x_id] = depth * MOVE_FORCE
    data.ctrl[p.act_z_id] = HIT_UP_FORCE if enable_hit_up and hit_up > 0.5 else 0.0

    if enable_jump_impulse and hit_up > 0.5 and is_grounded:
        data.qvel[p.slide_z_qvel] = JUMP_VEL


def reset_ball(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    idx: ModelIndices,
    pos: np.ndarray | None = None,
    vel: np.ndarray | None = None,
) -> None:
    ball_pos = BALL_INIT_POS if pos is None else np.asarray(pos, dtype=np.float64)
    ball_vel = BALL_INIT_VEL if vel is None else np.asarray(vel, dtype=np.float64)

    data.qpos[idx.ball_qpos:idx.ball_qpos + 3] = ball_pos
    data.qpos[idx.ball_qpos + 3:idx.ball_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[idx.ball_qvel:idx.ball_qvel + 6] = ball_vel
    mujoco.mj_forward(model, data)


def check_ground_touch(data: mujoco.MjData, idx: ModelIndices, state: SimState) -> bool:
    if time.time() < state.miss_cooldown_until:
        return False
    return float(data.qpos[idx.ball_qpos + 2]) < GROUND_Z_THRESHOLD
