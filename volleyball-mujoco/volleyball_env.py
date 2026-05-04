from __future__ import annotations

import os
from dataclasses import dataclass

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from volleyball_core import GROUND_TOL, GROUND_Z_THRESHOLD, HIT_UP_FORCE, ModelIndices, _build_indices, apply_action, reset_ball

NET_HEIGHT = 2.43
OBS_CLIP_POS = 10.0
OBS_CLIP_VEL = 30.0


@dataclass
class EpisodeStats:
    reward: float = 0.0
    touches: int = 0
    crossings: int = 0
    own_side_landings: int = 0
    opponent_side_landings: int = 0


class VolleyballEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 100}

    def __init__(
        self,
        xml_path: str = "assets/volleyball.xml",
        frame_skip: int = 5,
        render_mode: str | None = None,
        opponent_mode: str = "static",
        training_stage: int = 1,
        max_episode_steps: int = 1000,
        terminate_on_touch: bool = False,
        enable_jump_impulse: bool = False,
    ) -> None:
        super().__init__()
        here = os.path.dirname(os.path.abspath(__file__))
        self.xml_path = xml_path if os.path.isabs(xml_path) else os.path.join(here, xml_path)
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.opponent_mode = opponent_mode
        self.training_stage = training_stage
        self.max_episode_steps = max_episode_steps
        self.terminate_on_touch = terminate_on_touch
        self.enable_jump_impulse = enable_jump_impulse

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.idx: ModelIndices = _build_indices(self.model)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        low = np.full(18, -OBS_CLIP_VEL, dtype=np.float32)
        high = np.full(18, OBS_CLIP_VEL, dtype=np.float32)
        low[[0, 1, 2, 6, 7, 8, 12, 13, 14, 15, 16, 17]] = -OBS_CLIP_POS
        high[[0, 1, 2, 6, 7, 8, 12, 13, 14, 15, 16, 17]] = OBS_CLIP_POS
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._viewer = None
        self._np_random = np.random.default_rng()
        self._home_p2 = np.array([-3.5, 0.0], dtype=np.float64)
        self._episode_steps = 0
        self._stats = EpisodeStats()
        self._last_ball_x = 0.0
        self._last_touch_step = -1
        self._touch_ball_vz = 0.0
        self._prev_horiz_dist = 0.0
        self._touch_horiz_dist = 0.0

    def _set_player_pose(self, player, x: float, y: float, z: float) -> None:
        body_pos = self.model.body_pos[player.body_id]
        self.data.qpos[player.slide_y_qpos] = y - float(body_pos[1])
        self.data.qpos[player.slide_x_qpos] = x - float(body_pos[0])
        self.data.qpos[player.slide_z_qpos] = z - float(body_pos[2])
        self.data.qvel[player.slide_y_qvel] = 0.0
        self.data.qvel[player.slide_x_qvel] = 0.0
        self.data.qvel[player.slide_z_qvel] = 0.0

    def _get_obs(self) -> np.ndarray:
        p1_pos = self.data.xpos[self.idx.p1.body_id].copy()
        p1_vel = np.array(
            [
                self.data.qvel[self.idx.p1.slide_x_qvel],
                self.data.qvel[self.idx.p1.slide_y_qvel],
                self.data.qvel[self.idx.p1.slide_z_qvel],
            ],
            dtype=np.float64,
        )
        ball_pos = self.data.qpos[self.idx.ball_qpos:self.idx.ball_qpos + 3].copy()
        ball_vel = self.data.qvel[self.idx.ball_qvel:self.idx.ball_qvel + 3].copy()
        rel = ball_pos - p1_pos
        p2_pos = self.data.xpos[self.idx.p2.body_id].copy()

        obs = np.array(
            [
                p1_pos[0],
                p1_pos[1],
                p1_pos[2],
                p1_vel[0],
                p1_vel[1],
                p1_vel[2],
                ball_pos[0],
                ball_pos[1],
                ball_pos[2],
                ball_vel[0],
                ball_vel[1],
                ball_vel[2],
                rel[0],
                rel[1],
                rel[2],
                p2_pos[0],
                p2_pos[1],
                p2_pos[2],
            ],
            dtype=np.float32,
        )

        pos_idx = np.array([0, 1, 2, 6, 7, 8, 12, 13, 14, 15, 16, 17])
        vel_idx = np.array([3, 4, 5, 9, 10, 11])
        obs[pos_idx] = np.clip(obs[pos_idx], -OBS_CLIP_POS, OBS_CLIP_POS)
        obs[vel_idx] = np.clip(obs[vel_idx], -OBS_CLIP_VEL, OBS_CLIP_VEL)
        return obs

    def _sample_opponent_action(self) -> np.ndarray:
        if self.opponent_mode == "static":
            return np.zeros(3, dtype=np.float32)

        if self.opponent_mode == "random":
            if self._episode_steps % 10 == 0:
                self._random_action = self.action_space.sample()
            return getattr(self, "_random_action", np.zeros(3, dtype=np.float32))

        p2_pos = self.data.xpos[self.idx.p2.body_id]
        ball_pos = self.data.qpos[self.idx.ball_qpos:self.idx.ball_qpos + 3]
        ball_vel = self.data.qvel[self.idx.ball_qvel:self.idx.ball_qvel + 3]

        if ball_pos[0] < 0.0:
            target_x = float(np.clip(ball_pos[0], -7.5, -0.5))
            target_y = float(np.clip(ball_pos[1], -3.5, 3.5))
        else:
            target_x = float(self._home_p2[0])
            target_y = float(self._home_p2[1])

        err_x = target_x - float(p2_pos[0])
        err_y = target_y - float(p2_pos[1])
        depth = 0.0 if abs(err_x) < 0.15 else float(np.sign(err_x))
        lat = 0.0 if abs(err_y) < 0.15 else float(np.sign(err_y))

        horiz_dist = float(np.linalg.norm(ball_pos[:2] - p2_pos[:2]))
        jump = 1.0 if horiz_dist < 0.7 and 1.2 < ball_pos[2] < 2.7 and ball_vel[2] < 0.0 else 0.0
        return np.array([lat, depth, jump], dtype=np.float32)

    def _ball_touched_by_player(self, player_geom_name: str) -> bool:
        player_geom_id = self.idx.p1.geom_id if player_geom_name == "player_geom" else self.idx.p2.geom_id
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 == self.idx.ball_geom_id and geom2 == player_geom_id:
                return True
            if geom2 == self.idx.ball_geom_id and geom1 == player_geom_id:
                return True
        return False

    def _stage_reset_ball(self) -> None:
        random_y = float(self._np_random.uniform(-1.5, 1.5))
        if self.training_stage == 1:
            angle = float(self._np_random.uniform(0.0, 2.0 * np.pi))
            radius = float(self._np_random.uniform(0.45, 0.9))
            spawn_x = float(np.clip(3.5 + radius * np.cos(angle), 2.8, 4.8))
            spawn_y = float(np.clip(radius * np.sin(angle), -1.2, 1.2))
            pos = np.array([spawn_x, spawn_y, 3.2], dtype=np.float64)
            vel = np.array([0.0, 0.0, 2.2, 0.0, 0.0, 0.0], dtype=np.float64)
        elif self.training_stage == 2:
            spawn_x = float(self._np_random.uniform(2.9, 4.2))
            spawn_y = float(self._np_random.uniform(-0.9, 0.9))
            pos = np.array([spawn_x, spawn_y, 2.8], dtype=np.float64)
            vel = np.array(
                [
                    float(self._np_random.uniform(-0.3, 0.3)),
                    float(self._np_random.uniform(-0.2, 0.2)),
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float64,
            )
        elif self.training_stage == 3:
            pos = np.array([3.0, random_y, 2.0], dtype=np.float64)
            vel = np.array([-1.0, float(self._np_random.uniform(-0.5, 0.5)), 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            pos = np.array([3.5, 0.0, 3.0], dtype=np.float64)
            vel = np.zeros(6, dtype=np.float64)
        reset_ball(self.model, self.data, self.idx, pos=pos, vel=vel)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self._set_player_pose(self.idx.p1, x=3.5, y=0.0, z=0.6)
        self._set_player_pose(self.idx.p2, x=-3.5, y=0.0, z=0.6)
        self._stage_reset_ball()
        mujoco.mj_forward(self.model, self.data)

        self._episode_steps = 0
        self._stats = EpisodeStats()
        self._last_ball_x = float(self.data.qpos[self.idx.ball_qpos])
        self._last_touch_step = -1
        self._touch_ball_vz = 0.0
        self._touch_horiz_dist = 0.0
        p1_pos = self.data.xpos[self.idx.p1.body_id]
        ball_pos = self.data.qpos[self.idx.ball_qpos:self.idx.ball_qpos + 3]
        self._prev_horiz_dist = float(np.linalg.norm(ball_pos[:2] - p1_pos[:2]))

        return self._get_obs(), {}

    def _compute_reward(
        self,
        action: np.ndarray,
        p1_touch: bool,
        crossed_net: bool,
        ball_grounded: bool,
        landed_side: str | None,
    ) -> float:
        p1_pos = self.data.xpos[self.idx.p1.body_id]
        ball_pos = self.data.qpos[self.idx.ball_qpos:self.idx.ball_qpos + 3]
        ball_vel = self.data.qvel[self.idx.ball_qvel:self.idx.ball_qvel + 3]
        horiz_dist = float(np.linalg.norm(ball_pos[:2] - p1_pos[:2]))

        reward = 0.0
        if self.training_stage == 1:
            distance_improvement = self._prev_horiz_dist - horiz_dist
            reward += 0.08 * float(np.clip(distance_improvement, -0.5, 0.5))
            reward += -0.03 * horiz_dist
            reward += 0.2 if horiz_dist < 0.75 else 0.0
            reward += 1.5 if p1_touch else 0.0
            reward += -2.0 if ball_grounded else 0.0
            reward += -0.001
            return reward

        if self.training_stage == 2:
            reward += 0.01 if ball_pos[2] > GROUND_Z_THRESHOLD else 0.0
            reward += -0.015 * horiz_dist
            reward += 0.25 if horiz_dist < 0.8 else 0.0
            if action[2] > 0.5 and horiz_dist > 1.0:
                reward += -0.08
            reward += 2.5 if p1_touch else 0.0
            if p1_touch:
                reward += 0.75 if self._touch_horiz_dist < 0.6 else 0.0
                reward += 0.35 * max(self._touch_ball_vz, 0.0)
                reward += 0.75 if ball_pos[2] > 1.7 else 0.0
            reward += -3.0 if ball_grounded else 0.0
            reward += -0.001
            return reward

        if self.training_stage == 3:
            reward += 0.02 if ball_pos[2] > GROUND_Z_THRESHOLD else 0.0
            reward += -0.01 * horiz_dist
            reward += 2.0 if p1_touch else 0.0
            if p1_touch:
                reward += 0.2 * max(self._touch_ball_vz, 0.0)
                reward += 0.5 if ball_pos[2] > 1.5 else 0.0
                reward += 0.1 * max(-float(ball_vel[0]), 0.0)
            reward += 5.0 if crossed_net else 0.0
            reward += 10.0 if landed_side == "p2" else 0.0
            reward += -5.0 if landed_side == "p1" else 0.0
            reward += -3.0 if ball_grounded and landed_side is None else 0.0
            reward += -0.001
            return reward

        reward += 1.0 if p1_touch else 0.0
        reward += 5.0 if crossed_net else 0.0
        reward += 10.0 if landed_side == "p2" else 0.0
        reward += -10.0 if landed_side == "p1" else 0.0
        reward += 0.01 if not ball_grounded else 0.0
        reward += -0.001
        return reward

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        apply_action(
            self.data,
            action,
            self.idx.p1,
            enable_jump_impulse=self.enable_jump_impulse,
            enable_hit_up=True,
        )
        apply_action(
            self.data,
            self._sample_opponent_action(),
            self.idx.p2,
            enable_jump_impulse=self.enable_jump_impulse,
            enable_hit_up=True,
        )

        p1_touch = False
        crossed_net = False
        ball_grounded = False
        landed_side = None

        for _ in range(self.frame_skip):
            prev_ball_x = float(self.data.qpos[self.idx.ball_qpos])
            mujoco.mj_step(self.model, self.data)

            current_ball_x = float(self.data.qpos[self.idx.ball_qpos])
            current_ball_z = float(self.data.qpos[self.idx.ball_qpos + 2])

            if not p1_touch and self._ball_touched_by_player("player_geom"):
                p1_touch = True
                self._touch_ball_vz = float(self.data.qvel[self.idx.ball_qvel + 2])
                p1_pos = self.data.xpos[self.idx.p1.body_id]
                ball_pos = self.data.qpos[self.idx.ball_qpos:self.idx.ball_qpos + 3]
                self._touch_horiz_dist = float(np.linalg.norm(ball_pos[:2] - p1_pos[:2]))

            if prev_ball_x > 0.0 and current_ball_x < 0.0 and current_ball_z > NET_HEIGHT:
                crossed_net = True

            if current_ball_z < GROUND_Z_THRESHOLD:
                ball_grounded = True
                landed_side = "p1" if current_ball_x > 0.0 else "p2"
                break

        reward = self._compute_reward(action, p1_touch, crossed_net, ball_grounded, landed_side)
        self._episode_steps += 1
        self._stats.reward += reward
        self._stats.touches += int(p1_touch)
        self._stats.crossings += int(crossed_net)
        self._stats.own_side_landings += int(landed_side == "p1")
        self._stats.opponent_side_landings += int(landed_side == "p2")
        p1_pos = self.data.xpos[self.idx.p1.body_id]
        ball_pos = self.data.qpos[self.idx.ball_qpos:self.idx.ball_qpos + 3]
        self._prev_horiz_dist = float(np.linalg.norm(ball_pos[:2] - p1_pos[:2]))

        terminated = ball_grounded or (self.terminate_on_touch and p1_touch and self.training_stage in (1, 2))
        truncated = self._episode_steps >= self.max_episode_steps
        self._last_ball_x = float(self.data.qpos[self.idx.ball_qpos])

        info = {
            "p1_touch": p1_touch,
            "crossed_net": crossed_net,
            "ball_grounded": ball_grounded,
            "landed_side": landed_side,
            "ball_x": float(self.data.qpos[self.idx.ball_qpos]),
            "ball_z": float(self.data.qpos[self.idx.ball_qpos + 2]),
            "ball_vz": float(self.data.qvel[self.idx.ball_qvel + 2]),
            "touch_ball_vz": float(self._touch_ball_vz) if p1_touch else None,
            "touch_horiz_dist": float(self._touch_horiz_dist) if p1_touch else None,
            "hit_up_pressed": bool(action[2] > 0.5),
            "player_grounded": bool(float(self.data.qpos[self.idx.p1.slide_z_qpos]) < GROUND_TOL),
            "hit_up_active": bool(action[2] > 0.5),
            "hit_up_force": float(HIT_UP_FORCE if action[2] > 0.5 else 0.0),
            "episode_touches": self._stats.touches,
            "episode_crossings": self._stats.crossings,
            "episode_opponent_side_landings": self._stats.opponent_side_landings,
            "episode_own_side_landings": self._stats.own_side_landings,
            "episode_reward": self._stats.reward,
        }

        if terminated or truncated:
            info["episode"] = {
                "r": self._stats.reward,
                "l": self._episode_steps,
                "touches": self._stats.touches,
                "crossings": self._stats.crossings,
                "opponent_side_landings": self._stats.opponent_side_landings,
                "own_side_landings": self._stats.own_side_landings,
            }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return None
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.cam.lookat[:] = [0.0, 0.0, 1.4]
            self._viewer.cam.distance = 17.0
            self._viewer.cam.elevation = -15.0
            self._viewer.cam.azimuth = 90.0
        self._viewer.sync()
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
