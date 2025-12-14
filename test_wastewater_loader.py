import numpy as np
import torch
import args_new
from data_loader import load_steady_state, load_influent_profile, build_wastewater_datasets


def main():
    cfg = args_new.WASTEWATER_DATA
    steady, steady_mask = load_steady_state(cfg['steady_state_path'], cfg['expected_state_dim'])
    influent, influent_mask = load_influent_profile(cfg['influent_paths']['dry'], cfg['expected_influent_dim'])

    assert steady.shape[0] == cfg['expected_state_dim'], "steady state dimension mismatch"
    assert influent.shape[1] == cfg['expected_influent_dim'], "influent feature dimension mismatch"

    print(f"steady mask coverage: {steady_mask.mean():.2f}")
    print(f"influent coverage: {influent_mask.mean():.2f}")
    print(f"influent min/max: {np.nanmin(influent):.2f}/{np.nanmax(influent):.2f}")

    train_set, val_set, test_set, draw_set = build_wastewater_datasets(
        steady_path=cfg['steady_state_path'],
        influent_paths=cfg['influent_paths'],
        profile_key='dry',
        seq_length=cfg['seq_length'],
        prediction_horizons=cfg['prediction_horizons'],
        train_frac=cfg['train_frac'],
        val_frac=cfg['val_frac'],
        expected_state_dim=cfg['expected_state_dim'],
        expected_influent_dim=cfg['expected_influent_dim'],
        normalize=cfg['normalize'],
    )

    sample = train_set[0]
    x, u, mx, mu, horizon = sample
    assert x.shape[1] == cfg['expected_state_dim']
    assert u.shape[1] == cfg['expected_influent_dim']
    assert mx.min() >= 0 and mx.max() <= 1
    assert mu.min() >= 0 and mu.max() <= 1
    print(f"train sample shapes: x {tuple(x.shape)}, u {tuple(u.shape)}, horizon {horizon}")
    print(f"splits: train {len(train_set)}, val {len(val_set)}, test {len(test_set)}, draw {len(draw_set)}")

    # quick tensor conversion sanity
    _ = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=False)


if __name__ == "__main__":
    main()
