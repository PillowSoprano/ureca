"""Utility helpers for mapping normalized control vectors to MBR actuators.

The CasADi MBR simulator expects four physical inputs in the following order:

1. ``aeration`` (KLa for tank 3)
2. ``recirculation`` (KLa for tank 4)
3. ``wasting`` (``Q_r``)
4. ``influent_split`` (``Q_int``)

Controllers often operate on scaled/normalized action vectors (e.g., ``[-1, 1]``).
These helpers centralize the mapping, clipping, and scaling logic so every
controller interacts with the simulator using consistent bounds.
"""

from __future__ import annotations

import numpy as np


# Fixed ordering used by the simulator
MBR_ACTUATOR_INDEX = {
    "aeration": 0,
    "recirculation": 1,
    "wasting": 2,
    "influent_split": 3,
}


def denormalize_mbr_action(unit_action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Map a normalized ``[-1, 1]`` vector to simulator units and clip to bounds."""

    unit_action = np.asarray(unit_action, dtype=float)
    if unit_action.shape[-1] != 4:
        raise ValueError(f"MBR expects 4 actuators, got shape {unit_action.shape}")

    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    span = (high - low) / 2.0
    mid = (high + low) / 2.0
    scaled = mid + span * np.clip(unit_action, -1.0, 1.0)
    return np.clip(scaled, low, high)


def normalize_mbr_action(physical_action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Map simulator units to a normalized ``[-1, 1]`` representation."""

    physical_action = np.asarray(physical_action, dtype=float)
    if physical_action.shape[-1] != 4:
        raise ValueError(f"MBR expects 4 actuators, got shape {physical_action.shape}")

    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    span = (high - low) / 2.0
    mid = (high + low) / 2.0
    return np.clip((physical_action - mid) / np.clip(span, 1e-6, None), -1.0, 1.0)


def clip_mbr_action(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Safely clip a control vector using simulator bounds."""

    action = np.asarray(action, dtype=float)
    if action.shape[-1] != 4:
        raise ValueError(f"MBR expects 4 actuators, got shape {action.shape}")
    return np.clip(action, low, high)


def summarize_actuators(actions: np.ndarray, low: np.ndarray, high: np.ndarray) -> dict:
    """Return min/mean/max utilization for each actuator for logging."""

    actions = np.asarray(actions, dtype=float)
    if actions.ndim == 1:
        actions = actions[None, :]
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    span = np.clip(high - low, 1e-6, None)
    utilization = (actions - low) / span

    stats = {}
    for name, idx in MBR_ACTUATOR_INDEX.items():
        values = utilization[:, idx]
        stats[name] = {
            "min": float(values.min()),
            "mean": float(values.mean()),
            "max": float(values.max()),
            "saturation_pct": float(np.mean((actions[:, idx] <= low[idx]) | (actions[:, idx] >= high[idx])) * 100.0),
        }
    return stats

