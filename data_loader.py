"""Utilities for loading wastewater steady-state and influent datasets."""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, random_split

try:
    from waste_water_system import waste_water_system
except Exception:
    waste_water_system = None

Array = np.ndarray


def _pad_or_trim(vec: Array, target_len: Optional[int], fill_value: float = np.nan) -> Array:
    vec = np.asarray(vec, dtype=np.float32).flatten()
    if target_len is None:
        return vec
    if vec.size == target_len:
        return vec
    padded = np.full((target_len,), fill_value, dtype=np.float32)
    copy_len = min(vec.size, target_len)
    padded[:copy_len] = vec[:copy_len]
    return padded


def load_steady_state(path: str, expected_dim: Optional[int] = 156) -> Tuple[Array, Array]:
    """Load steady-state vector with padding and mask."""
    values = np.loadtxt(path, dtype=np.float32).reshape(-1)
    if expected_dim is not None:
        values = _pad_or_trim(values, expected_dim)
    mask = np.isfinite(values).astype(np.float32)
    values = np.nan_to_num(values, nan=0.0)
    return values, mask


def load_influent_profile(path: str, expected_features: int = 15) -> Tuple[Array, Array]:
    """Parse influent profile tables such as ``Inf_dry_2006.txt``."""
    rows: List[List[float]] = []
    with open(path, "r") as f:
        for raw in f:
            parts = [p for p in raw.strip().split() if p]
            if len(parts) < expected_features:
                continue
            try:
                row = [float(p) for p in parts[:expected_features]]
            except ValueError:
                continue
            rows.append(row)
    data = np.asarray(rows, dtype=np.float32)
    mask = np.isfinite(data).astype(np.float32)
    data = np.nan_to_num(data, nan=0.0)
    return data, mask


def compute_normalization(data: Array, mask: Array) -> Tuple[Array, Array]:
    valid = mask.astype(bool)
    weighted = np.where(valid, data, 0.0)
    count = valid.sum(axis=(0, 1))
    denom = np.maximum(count, 1.0)
    mean = weighted.sum(axis=(0, 1)) / denom
    variance = ((np.where(valid, data, 0.0) - mean) ** 2 * valid).sum(axis=(0, 1)) / denom
    std = np.sqrt(variance)
    std[std == 0] = 1.0
    mean[count == 0] = 0.0
    std[count == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


class WasteWaterDataset(Dataset):
    """Dataset that aligns state and influent sequences with masks."""

    def __init__(
        self,
        steady_state: Array,
        steady_mask: Array,
        influent: Array,
        influent_mask: Array,
        seq_length: int,
        prediction_horizons: Sequence[int],
        normalize: bool = True,
        stats: Optional[Tuple[Array, Array, Array, Array]] = None,
    ) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.prediction_horizons = list(prediction_horizons)
        self.influent = influent.astype(np.float32)
        self.influent_mask = influent_mask.astype(np.float32)
        self.state_dim = steady_state.size
        self.influent_dim = influent.shape[1]

        self.steady = steady_state.astype(np.float32)
        self.steady_mask = steady_mask.astype(np.float32)

        if normalize:
            if stats is None:
                ss_mu, ss_sigma = compute_normalization(
                    np.tile(self.steady[None, None, :], (1, 1, 1)),
                    np.tile(self.steady_mask[None, None, :], (1, 1, 1)),
                )
                inf_mu, inf_sigma = compute_normalization(
                    self.influent[None, :, :], self.influent_mask[None, :, :]
                )
            else:
                ss_mu, ss_sigma, inf_mu, inf_sigma = stats
        else:
            ss_mu = np.zeros((self.state_dim,), dtype=np.float32)
            ss_sigma = np.ones((self.state_dim,), dtype=np.float32)
            inf_mu = np.zeros((self.influent_dim,), dtype=np.float32)
            inf_sigma = np.ones((self.influent_dim,), dtype=np.float32)
        self.stats = (ss_mu, ss_sigma, inf_mu, inf_sigma)

        self.samples: List[Tuple[int, int]] = []
        total = self.influent.shape[0]
        for horizon in self.prediction_horizons:
            max_start = total - (seq_length + horizon)
            for start in range(max(0, max_start + 1)):
                self.samples.append((start, horizon))

    def __len__(self) -> int:
        return len(self.samples)

    def _build_state(self, length: int) -> Tuple[Array, Array]:
        x = np.tile(self.steady[None, :], (length, 1))
        m = np.tile(self.steady_mask[None, :], (length, 1))
        return x, m

    def __getitem__(self, idx: int):
        start, horizon = self.samples[idx]
        total_len = self.seq_length + horizon
        u_slice = self.influent[start : start + total_len]
        m_u = self.influent_mask[start : start + total_len]
        x, m_x = self._build_state(total_len)

        ss_mu, ss_sigma, inf_mu, inf_sigma = self.stats
        x = (x - ss_mu) / ss_sigma
        u_slice = (u_slice - inf_mu) / inf_sigma

        return (
            torch.from_numpy(x),
            torch.from_numpy(u_slice),
            torch.from_numpy(m_x),
            torch.from_numpy(m_u),
            horizon,
        )


def build_wastewater_datasets(
    steady_path: str,
    influent_paths: Dict[str, str],
    profile_key: str,
    seq_length: int,
    prediction_horizons: Sequence[int],
    train_frac: float,
    val_frac: float,
    expected_state_dim: int = 156,
    expected_influent_dim: int = 15,
    normalize: bool = True,
    seed: int = 1,
) -> Tuple[WasteWaterDataset, WasteWaterDataset, WasteWaterDataset, WasteWaterDataset]:
    steady, steady_mask = load_steady_state(steady_path, expected_state_dim)
    influent, infl_mask = load_influent_profile(influent_paths[profile_key], expected_influent_dim)

    dataset = WasteWaterDataset(
        steady,
        steady_mask,
        influent,
        infl_mask,
        seq_length=seq_length,
        prediction_horizons=prediction_horizons,
        normalize=normalize,
    )

    total = len(dataset)
    n_train = int(total * train_frac)
    n_val = int(total * val_frac)
    n_test = max(total - n_train - n_val, 0)
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=generator)

    draw_set = WasteWaterDataset(
        steady,
        steady_mask,
        influent,
        infl_mask,
        seq_length=seq_length,
        prediction_horizons=[prediction_horizons[-1]],
        normalize=normalize,
        stats=dataset.stats,
    )
    return train_set, val_set, test_set, draw_set


def simulate_with_wastewater_model(horizon: int):
    if waste_water_system is None:
        return None
    env = waste_water_system()
    env.reset()
    states = []
    actions = []
    for _ in range(horizon):
        action = env.get_action()
        _, _, done, _ = env.step(action)
        states.append(env.x)
        actions.append(action)
        if done:
            break
    return [np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.float32)]
