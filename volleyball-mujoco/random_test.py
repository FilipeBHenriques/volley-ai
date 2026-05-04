from __future__ import annotations

import math

from volleyball_env import VolleyballEnv


def main() -> None:
    env = VolleyballEnv(frame_skip=5, opponent_mode="static", training_stage=1, max_episode_steps=200)
    episode_rewards = []

    for episode in range(10):
        obs, _ = env.reset(seed=episode)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            assert obs.shape == (18,)
            assert all(math.isfinite(float(x)) for x in obs)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert math.isfinite(float(reward))
            assert all(math.isfinite(float(x)) for x in obs)
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated

        episode_rewards.append(total_reward)
        print(
            f"episode={episode + 1} steps={steps} reward={total_reward:.3f} "
            f"touches={info['episode_touches']} crossings={info['episode_crossings']} "
            f"own_landings={info['episode_own_side_landings']} opp_landings={info['episode_opponent_side_landings']}"
        )

    avg_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"average_reward={avg_reward:.3f}")
    env.close()


if __name__ == "__main__":
    main()
