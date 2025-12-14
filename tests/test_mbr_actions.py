import numpy as np

from mbr_actions import clip_mbr_action, denormalize_mbr_action
from sim_interface import MBRTrajectorySimulator


def test_denormalize_respects_bounds():
    sim = MBRTrajectorySimulator()
    low, high = sim.env.action_low, sim.env.action_high
    unit_actions = np.random.uniform(-1.2, 1.2, size=(8, 4))
    physical = denormalize_mbr_action(unit_actions, low, high)
    assert np.all(physical >= low)
    assert np.all(physical <= high)


def test_rollout_clips_actions_and_remains_finite():
    sim = MBRTrajectorySimulator(step_size=1)
    low, high = sim.env.action_low, sim.env.action_high

    # Start from default state and apply near-boundary normalized actions
    actions_unit = np.linspace(-1, 1, num=4).reshape(1, 4)
    actions = np.stack([denormalize_mbr_action(actions_unit, low, high) for _ in range(3)], axis=0).squeeze(1)
    actions[1] = clip_mbr_action(actions[1] * 1.2, low, high)

    rollout = sim.rollout(actions[None, ...])
    states = rollout["states"][0]

    assert np.isfinite(states).all()
    assert np.isfinite(rollout["costs"]).all()
    assert np.all(actions >= low)
    assert np.all(actions <= high)
