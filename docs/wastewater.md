# Wastewater workflow guide

This guide summarizes how to prepare the influent datasets, pick configuration flags, and run the training, hybrid rollouts, and resilience evaluation utilities for the wastewater benchmark.

## Expected files and folders

Place the wastewater data in the repository root (or `envs/` with the same names) before running any scripts:

```
Inf_dry_2006.txt   # dry-weather influent matrix shaped (1345, 15)
inf_rain.txt       # rain scenario, shaped (15, 1345); transpose to (1345, 15) if needed
inf_strm.txt       # storm scenario, shaped (15, 1345); transpose to (1345, 15) if needed
ss_open.txt        # 145-element steady-state vector used to seed rollouts
```

Validate the files before training:

```bash
python scripts/verify_wastewater_data.py
```

During training and evaluation you will also see automatically created folders:

```
save_model/<method>/waste_water/   # checkpoints, normalizers, optimizer state
log/...                            # training and sweep logs
results/resilience/<scenario>/     # KPI tables and rollout plots
```

## Configuration highlights

Most defaults live in [`args_new.py`](../args_new.py). For wastewater training and control, key flags include:

- `batch_size`, `pred_horizon`, `old_horizon`: sequence length and batching for `ReplayMemory`.
- `training_mode`: set to `hybrid` to blend simulator rollouts from `ss_open.txt` (see `train.build_hybrid_rollouts`).
- `sim_rollout_length`, `sim_batches`, `sim_noise_scale`, `sim_fouling_perturb`: control the synthetic trajectories mixed in during hybrid training.
- `lr_scheduler`, `scheduler_step`, `scheduler_gamma`, `scheduler_min_lr`: learning-rate scheduling.
- `z_dim`, `h_dim`, `alpha`, `beta`, `grad_clip`, `use_action`: KoVAE-specific knobs.

For resilience evaluation, controller checkpoints must live under `save_model/<method>/waste_water/` with accompanying `shift_*.txt` scaling files.

## Training commands

Standard data-driven training:

```bash
# Train a KoVAE model (replace with mamko/dko/mlp as needed)
python train.py kovae cartpole
```

Hybrid mode (adds simulated rollouts seeded by `ss_open.txt`):

```bash
# Set `training_mode = "hybrid"` inside `args_new.py` before launching
python train.py kovae cartpole
```

> **Tip:** The command-line interface of `train.py` expects `method` and `model` positional arguments. If you introduce a wastewater entry to `ENV_PARAMS`, call it with `python train.py kovae waste_water` so the saved checkpoints match the expected `save_model/<method>/waste_water/` layout.

GPU/CPU tips:

- Set `CUDA_VISIBLE_DEVICES=""` to force CPU training when GPUs are unavailable.
- Reduce `batch_size` and sequence horizons on CPU-only machines to keep memory usage manageable.
- When using GPUs, keep `grad_clip` enabled and consider `lr_scheduler=cosine` to stabilize long runs.

## Resilience evaluation

Once checkpoints exist under `save_model/<method>/waste_water/`, run the batch evaluator:

```bash
python resilience_eval.py --scenario dry --methods mamko kovae --seeds 0 1 2 \
  --max-steps 288 --output results/resilience
```

This generates rollout plots, KPI CSVs, bar charts, and a Markdown summary in `results/resilience/<scenario>/`.

To view plots after the run, open the generated PNGs in the output folder (e.g., `mamko_seed0.png`, `violations.png`).

## Quick plotting without evaluation

If you only need simulator-driven plots for the saved controllers, reuse the rollout plot step:

```bash
python resilience_eval.py --scenario storm --methods kovae --seeds 0 \
  --max-steps 288 --output results/resilience_only_plots
```

The script will still compute KPIs but you can ignore the CSV and focus on the figure artifacts.
