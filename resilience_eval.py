"""Resilience evaluation utilities for wastewater rollouts.

This module provides controlled rollouts under different inflow
scenarios (dry, rain, storm) and computes resilience KPIs for
MamKO and KoVAE controllers.  The script can be executed directly
for batch evaluation across random seeds.
"""
import argparse
import csv
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import args_new as new_args
import logger
from controller import Upper_MPC_DKO, Upper_MPC_mamba
from waste_water_system import waste_water_system
from MamKO import Koopman_Desko as MamKOModel
from kovae_model import Koopman_Desko as KoVAEModel

SCENARIO_FILES = {
    "dry": "Inf_dry_2006.txt",
    "rain": "inf_rain.txt",
    "storm": "inf_strm.txt",
}


@dataclass
class RolloutLog:
    cost: List[float]
    effluent_quality: List[float]
    aeration_energy: List[float]
    pumping_energy: List[float]
    sludge_production: List[float]
    maintenance_energy: List[float]
    control_effort: List[float]
    tracking_error: List[float]


@dataclass
class KPIResult:
    method: str
    seed: int
    scenario: str
    effluent_violations: int
    recovery_time: int
    avg_control_effort: float
    fouling_indicator: float


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def load_inflow(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    return data[:, 14][:, np.newaxis], data[:, 1:14]


def build_env(scenario: str, base_path: Path) -> waste_water_system:
    env = waste_water_system()
    inflow_path = base_path / SCENARIO_FILES[scenario]
    Q0, Z0 = load_inflow(inflow_path)
    env.Q0 = Q0
    env.Z0 = Z0
    env.t = 0
    env.state_buffer.reset()
    return env


# ---------------------------------------------------------------------------
# Controller loading
# ---------------------------------------------------------------------------

def _set_common_args(env: waste_water_system) -> Dict:
    args = dict(new_args.args)
    args.update(
        {
            "state_dim": env.observation_space.shape[0],
            "act_dim": env.action_space.shape[0],
            "control": True,
            "env_name": "waste_water",
            "pred_horizon": 15,
            "control_horizon": 15,
            "latent_dim": 16,
            "disturbance": 0,
            "apply_state_constraints": False,
            "apply_action_constraints": True,
            "reference": env.xs,
            "reference_": env.xs,
            "s_bound_low": env.observation_space.low,
            "s_bound_high": env.observation_space.high,
            "a_bound_low": env.action_space.low,
            "a_bound_high": env.action_space.high,
        }
    )
    return args


def load_controller(method: str, env: waste_water_system, save_root: Path):
    args = _set_common_args(env)
    method_lower = method.lower()
    fold_path = save_root / method_lower / "waste_water"
    fold_path.mkdir(parents=True, exist_ok=True)

    args.update(
        {
            "save_model_path": str(fold_path / "model.pt"),
            "save_opti_path": str(fold_path / "opti.pt"),
            "shift_x": str(fold_path / "shift_x.txt"),
            "scale_x": str(fold_path / "scale_x.txt"),
            "shift_u": str(fold_path / "shift_u.txt"),
            "scale_u": str(fold_path / "scale_u.txt"),
        }
    )

    if method_lower == "mamko":
        model = MamKOModel(args)
        model.shift = np.loadtxt(args["shift_x"])
        model.shift_u = np.loadtxt(args["shift_u"])
        model.scale = np.loadtxt(args["scale_x"])
        model.scale_u = np.loadtxt(args["scale_u"])
        model.parameter_restore(args)
        controller = Upper_MPC_mamba(
            model,
            args,
            np.eye(env.state_tracking.shape[0]),
            np.eye(env.action_space.shape[0]),
            np.eye(env.state_tracking.shape[0]),
        )
    elif method_lower == "kovae":
        model = KoVAEModel(args)
        model.shift = np.loadtxt(args["shift_x"])
        model.shift_u = np.loadtxt(args["shift_u"])
        model.scale = np.loadtxt(args["scale_x"])
        model.scale_u = np.loadtxt(args["scale_u"])
        model.parameter_restore(args)
        model.shift_ = [model.shift, model.scale, model.shift_u, model.scale_u]
        controller = Upper_MPC_DKO(
            model,
            args,
            np.eye(env.state_tracking.shape[0]),
            np.eye(env.action_space.shape[0]),
            np.eye(env.state_tracking.shape[0]),
        )
    else:
        raise ValueError(f"Unknown method {method}")

    controller.restore()
    return controller


# ---------------------------------------------------------------------------
# Rollout & KPI calculation
# ---------------------------------------------------------------------------

def run_rollout(controller, env: waste_water_system, seed: int, max_steps: int) -> RolloutLog:
    np.random.seed(seed)
    env.seed(seed)
    obs = env.reset()
    cost_log: List[float] = []
    eq_log: List[float] = []
    ae_log: List[float] = []
    pe_log: List[float] = []
    sp_log: List[float] = []
    me_log: List[float] = []
    effort_log: List[float] = []
    tracking_log: List[float] = []

    for t in range(max_steps):
        act = controller.choose_action(obs, env.xs)
        obs, cost, done, info = env.step(act)
        eq, ae, pe, sp, me = env.cost_every
        cost_log.append(cost)
        eq_log.append(eq)
        ae_log.append(ae)
        pe_log.append(pe)
        sp_log.append(sp)
        me_log.append(me)
        effort_log.append(float(np.linalg.norm(act)))
        tracking_log.append(float(np.linalg.norm(env.x[:2] - env.xs)))
        if done:
            break

    return RolloutLog(
        cost=cost_log,
        effluent_quality=eq_log,
        aeration_energy=ae_log,
        pumping_energy=pe_log,
        sludge_production=sp_log,
        maintenance_energy=me_log,
        control_effort=effort_log,
        tracking_error=tracking_log,
    )


def compute_kpis(log: RolloutLog, method: str, seed: int, scenario: str, violation_threshold: float) -> KPIResult:
    effluent_violations = int(np.sum(np.array(log.effluent_quality) > violation_threshold))

    recovery_time = len(log.tracking_error)
    for idx, err in enumerate(log.tracking_error):
        if err < 0.05:
            recovery_time = idx
            break

    avg_effort = float(np.mean(log.control_effort)) if log.control_effort else 0.0
    fouling_indicator = float(np.mean(log.sludge_production)) if log.sludge_production else 0.0

    return KPIResult(
        method=method,
        seed=seed,
        scenario=scenario,
        effluent_violations=effluent_violations,
        recovery_time=recovery_time,
        avg_control_effort=avg_effort,
        fouling_indicator=fouling_indicator,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_rollout_plot(log: RolloutLog, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(log.effluent_quality, label="EQ")
    axes[0].set_ylabel("Effluent quality")
    axes[0].legend()

    axes[1].plot(log.control_effort, label="Control effort")
    axes[1].set_ylabel("||u||")
    axes[1].legend()

    axes[2].plot(log.tracking_error, label="Tracking error")
    axes[2].set_ylabel("||x-x*||")
    axes[2].set_xlabel("Time step")
    axes[2].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_kpi_csv(results: List[KPIResult], out_path: Path) -> None:
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "seed",
                "scenario",
                "effluent_violations",
                "recovery_time",
                "avg_control_effort",
                "fouling_indicator",
            ],
        )
        writer.writeheader()
        for res in results:
            writer.writerow(asdict(res))


def save_bar_plot(results: List[KPIResult], metric: str, out_path: Path) -> None:
    methods = sorted({r.method for r in results})
    values = []
    for m in methods:
        vals = [getattr(r, metric) for r in results if r.method == m]
        values.append(np.mean(vals))

    plt.figure(figsize=(6, 4))
    plt.bar(methods, values)
    plt.ylabel(metric.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_report(results: List[KPIResult], out_dir: Path, scenario: str) -> None:
    report_path = out_dir / "resilience_report.md"
    lines = [f"# Resilience summary ({scenario})", "", "| Method | Effluent violations | Recovery time | Avg control effort | Fouling indicator |", "|---|---|---|---|---|"]
    for r in results:
        lines.append(
            f"| {r.method} (seed {r.seed}) | {r.effluent_violations} | {r.recovery_time} | {r.avg_control_effort:.3f} | {r.fouling_indicator:.3f} |"
        )
    lines.append("")
    lines.append(f"![Effluent violations]({(out_dir / 'violations.png').name})")
    lines.append(f"![Recovery time]({(out_dir / 'recovery_time.png').name})")
    lines.append(f"![Control effort]({(out_dir / 'avg_control_effort.png').name})")
    lines.append(f"![Fouling indicator]({(out_dir / 'fouling_indicator.png').name})")
    report_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Resilience evaluation for MamKO vs. KoVAE")
    parser.add_argument("--scenario", choices=list(SCENARIO_FILES.keys()), default="dry")
    parser.add_argument("--methods", nargs="+", default=["mamko", "kovae"], help="Controllers to evaluate")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-steps", type=int, default=288, help="Max rollout steps")
    parser.add_argument("--violation-threshold", type=float, default=0.5, help="Effluent quality violation threshold")
    parser.add_argument("--output", type=Path, default=Path("results/resilience"))
    parser.add_argument("--save-root", type=Path, default=Path("save_model"), help="Location of saved controllers")
    return parser.parse_args()


def main():
    args = parse_args()
    base_path = Path(__file__).resolve().parent
    out_dir = args.output / args.scenario
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.configure(dir=str(out_dir), format_strs=["csv", "stdout"])

    kpi_results: List[KPIResult] = []
    for method in args.methods:
        for seed in args.seeds:
            env = build_env(args.scenario, base_path)
            controller = load_controller(method, env, args.save_root)
            rollout = run_rollout(controller, env, seed, args.max_steps)
            kpis = compute_kpis(rollout, method, seed, args.scenario, args.violation_threshold)
            kpi_results.append(kpis)

            logger.logkv("method", method)
            logger.logkv("seed", seed)
            logger.logkv("effluent_violations", kpis.effluent_violations)
            logger.logkv("recovery_time", kpis.recovery_time)
            logger.logkv("avg_control_effort", kpis.avg_control_effort)
            logger.logkv("fouling_indicator", kpis.fouling_indicator)
            logger.dumpkvs()

            save_rollout_plot(rollout, out_dir / f"{method}_seed{seed}.png", f"{method} seed {seed}")

    save_kpi_csv(kpi_results, out_dir / "resilience_kpis.csv")
    for metric in ["effluent_violations", "recovery_time", "avg_control_effort", "fouling_indicator"]:
        save_bar_plot(kpi_results, metric, out_dir / f"{metric}.png")
    save_report(kpi_results, out_dir, args.scenario)


if __name__ == "__main__":
    main()
