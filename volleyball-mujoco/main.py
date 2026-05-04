"""
Manual-play entry point for the MuJoCo volleyball simulator.
"""

from __future__ import annotations

import math
import os
import time

import mujoco
import mujoco.viewer
import numpy as np

from volleyball_core import ModelIndices, SimState, _build_indices, apply_action, check_ground_touch, reset_ball

_HERE = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(_HERE, "assets", "volleyball.xml")
MISS_COOLDOWN = 0.5

KEY_W, KEY_S, KEY_A, KEY_D = 87, 83, 65, 68
KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262
KEY_SPACE, KEY_R = 32, 82

AI_DEAD_ZONE = 0.15
AI_HOME_X = -3.5
AI_HOME_Y = 0.0
AI_NET_GUARD = -0.5
AI_REACH_RAD = 0.6
AI_REACH_HI = 2.6
AI_REACH_LO = 1.4


class Controller:
    def __init__(self) -> None:
        self._held: set[int] = set()
        self._jump_pending = False
        self._reset_pending = False

    def on_key(self, *args: int) -> None:
        key = args[0]
        if len(args) == 4:
            action = args[2]
            if action in (1, 2):
                self._held.add(key)
            else:
                self._held.discard(key)
        else:
            if key in self._held:
                self._held.discard(key)
            else:
                self._held.add(key)

        is_press = len(args) < 4 or args[2] == 1
        if is_press and key == KEY_SPACE:
            self._jump_pending = True
            self._held.discard(key)
        if is_press and key == KEY_R:
            self._reset_pending = True
            self._held.discard(key)

    def poll_action(self) -> np.ndarray:
        lat = 0.0
        depth = 0.0

        if KEY_A in self._held or KEY_LEFT in self._held:
            lat -= 1.0
        if KEY_D in self._held or KEY_RIGHT in self._held:
            lat += 1.0
        if KEY_W in self._held or KEY_UP in self._held:
            depth -= 1.0
        if KEY_S in self._held or KEY_DOWN in self._held:
            depth += 1.0

        jump = 1.0 if self._jump_pending else 0.0
        self._jump_pending = False
        return np.array([lat, depth, jump], dtype=np.float32)

    def consume_reset(self) -> bool:
        flag = self._reset_pending
        self._reset_pending = False
        return flag


def ai_policy(data: mujoco.MjData, idx: ModelIndices) -> np.ndarray:
    p2_pos = data.xpos[idx.p2.body_id]
    ball_pos = data.xpos[idx.ball_body_id]
    ball_vz = float(data.qvel[idx.ball_qvel + 2])

    if ball_pos[0] < 0.5:
        target_x = max(-7.5, min(AI_NET_GUARD, float(ball_pos[0])))
        target_y = float(np.clip(ball_pos[1], -3.5, 3.5))
    else:
        target_x = AI_HOME_X
        target_y = AI_HOME_Y

    err_x = target_x - float(p2_pos[0])
    err_y = target_y - float(p2_pos[1])

    depth = 0.0
    if err_x > AI_DEAD_ZONE:
        depth = 1.0
    elif err_x < -AI_DEAD_ZONE:
        depth = -1.0

    lat = 0.0
    if err_y > AI_DEAD_ZONE:
        lat = 1.0
    elif err_y < -AI_DEAD_ZONE:
        lat = -1.0

    jump = 0.0
    horiz_dist = math.hypot(float(ball_pos[0] - p2_pos[0]), float(ball_pos[1] - p2_pos[1]))
    if horiz_dist < AI_REACH_RAD and AI_REACH_LO < ball_pos[2] < AI_REACH_HI and ball_vz < 0.0:
        jump = 1.0

    return np.array([lat, depth, jump], dtype=np.float32)


def run() -> None:
    if not os.path.isfile(XML_PATH):
        raise FileNotFoundError(f"Cannot find XML at: {XML_PATH}")

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    idx = _build_indices(model)
    state = SimState()
    controller = Controller()

    reset_ball(model, data, idx)

    with mujoco.viewer.launch_passive(model, data, key_callback=controller.on_key) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 1.4]
        viewer.cam.distance = 17.0
        viewer.cam.elevation = -15.0
        viewer.cam.azimuth = 90.0

        print("Viewer open. WASD/arrows + Space control Player 1.")
        print("Player 2 remains a lightweight scripted opponent. Press R to reset.")

        while viewer.is_running():
            t0 = time.perf_counter()

            if controller.consume_reset():
                reset_ball(model, data, idx)

            apply_action(data, controller.poll_action(), idx.p1)
            apply_action(data, ai_policy(data, idx), idx.p2)
            mujoco.mj_step(model, data)

            if check_ground_touch(data, idx, state):
                state.misses += 1
                state.miss_cooldown_until = time.time() + MISS_COOLDOWN
                print(f"MISS! total = {state.misses}")
                reset_ball(model, data, idx)

            viewer.sync()
            remaining = model.opt.timestep - (time.perf_counter() - t0)
            if remaining > 0.0:
                time.sleep(remaining)


if __name__ == "__main__":
    run()
