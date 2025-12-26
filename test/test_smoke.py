import numpy as np
from shelem_gym import shelem_v0

def test_env_runs_to_completion():
    env = shelem_v0()
    env.reset()

    step_count = 0
    for agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()

        if terminated or truncated:
            action = None
        else:
            mask = obs["action_mask"]
            legal = np.flatnonzero(mask)
            assert len(legal) > 0, "No legal actions for active agent"
            action = int(legal[0])  # dumb but valid

        env.step(action)
        step_count += 1

    # 4 players × 12 cards = 48 steps
    assert step_count == 48, f"Expected 48 steps, got {step_count}"
