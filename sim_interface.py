"""Lightweight wrapper around the CasADi-based MBR simulator.

This module exposes simple ``reset``/``step`` helpers and a batched
rollout utility so training code can request multiple simulations with
4-input control signals without dealing with the gym.Env interface.
"""

from __future__ import annotations

import numpy as np

from mbr_casadi_4inputs_limu import MBRGymEnv


class MBRTrajectorySimulator:
    """Utility for running MBR rollouts with 4-input control vectors.

    Parameters
    ----------
    step_size : int
        Step size multiplier when sampling controls. ``1`` keeps the
        native simulator resolution.
    noise_scale : float
        Gaussian noise scale applied to actions to emulate actuator
        uncertainty.
    fouling_perturb : float
        Additive perturbation on the fouling accumulator ``Rp`` per step
        to mimic fouling variability.
    """

    def __init__(self, step_size: int = 1, noise_scale: float = 0.0, fouling_perturb: float = 0.0):
        self.env = MBRGymEnv()
        self.step_size = max(1, int(step_size))
        self.noise_scale = float(noise_scale)
        self.fouling_perturb = float(fouling_perturb)
        self._disturbance = self.env.disturbance

    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        """Reset the simulator and optionally overwrite the initial state."""

        base_state = self.env.reset()
        if state is not None:
            if state.shape != base_state.shape:
                raise ValueError(f"Provided state has shape {state.shape}, expected {base_state.shape}")
            self.env.state = state.copy()
            base_state = self.env.state
        self._disturbance = self.env.disturbance
        return base_state

    def step(self, action: np.ndarray, disturbance_index: int) -> tuple[np.ndarray, float]:
        """Single simulation step returning next state and stage cost."""

        if action.shape[-1] != 4:
            raise ValueError(f"Expected 4 control inputs, got shape {action.shape}")

        noisy_action = action + np.random.normal(0.0, self.noise_scale, size=action.shape)
        noisy_action = np.clip(noisy_action, self.env.action_low, self.env.action_high)
        next_state, cost, *_ = self.env.step(noisy_action, disturbance_index)
        self.env.Rp += self.fouling_perturb
        return next_state, float(cost)

    def rollout(self, actions: np.ndarray, start_state: np.ndarray | None = None) -> dict:
        """Run a batched rollout.

        Parameters
        ----------
        actions : np.ndarray
            Array shaped ``[B, T, 4]`` of control signals.
        start_state : np.ndarray, optional
            Initial state. If ``None`` the environment default reset is
            used for each batch element.

        Returns
        -------
        dict
            ``{"states": states, "costs": costs}`` where ``states`` has
            shape ``[B, T+1, state_dim]`` and ``costs`` has shape
            ``[B, T]``.
        """

        if actions.ndim != 3 or actions.shape[-1] != 4:
            raise ValueError("actions must have shape [B, T, 4]")

        B, T, _ = actions.shape
        all_states = []
        all_costs = []

        for b in range(B):
            self.reset(start_state)
            states = [self.env.state.copy()]
            costs = []
            for t in range(T):
                # respect step_size by repeating the same action
                for _ in range(self.step_size):
                    s, c = self.step(actions[b, t], disturbance_index=min(t, self._disturbance.shape[0]-1))
                    states.append(s.copy())
                    costs.append(c)
            all_states.append(np.stack(states, axis=0))
            all_costs.append(np.array(costs, dtype=np.float32))

        return {"states": np.stack(all_states, axis=0), "costs": np.stack(all_costs, axis=0)}

