# compare_mamko_kovae.py  (fixed: single dataset, log plots, RMSE table)
import os, numpy as np, torch, matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader
import args_new as new_args
from replay_fouling import ReplayMemory
from data_loader import build_wastewater_datasets
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
plt.rcParams["figure.dpi"]=150

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='sim', choices=['sim', 'wastewater'])
parser.add_argument('--influent-profile', type=str, default='dry')
args_cli = parser.parse_args()

MODEL = "wastewater" if args_cli.dataset == 'wastewater' else "cartpole"
METHODS = ["kovae","mamba"]
OUTDIR = f"loss/compare/{MODEL}"

def get_args(method):
    args = dict(new_args.args, **new_args.ENV_PARAMS.get(MODEL, {}))
    args["env"] = MODEL
    args["method"] = method
    fold = f"save_model/{method}/{MODEL}"
    args["save_model_path"] = f"{fold}/model.pt"
    args["save_opti_path"]  = f"{fold}/opti.pt"
    args["shift_x"]         = f"{fold}/shift_x.txt"
    args["scale_x"]         = f"{fold}/scale_x.txt"
    args["shift_u"]         = f"{fold}/shift_u.txt"
    args["scale_u"]         = f"{fold}/scale_u.txt"
    args["control"] = False
    return args

def load_model(method, args):
    if method == "kovae":
        from kovae_model import Koopman_Desko
    elif method == "mamba":
        from MamKO import Koopman_Desko
    else:
        raise ValueError
    m = Koopman_Desko(args); m.parameter_restore(args); return m

def rmse_curve(model, test_draw, args, H=100):
    dl = DataLoader(test_draw, batch_size=1, shuffle=False, drop_last=False)
    SSE = np.zeros(H, dtype=np.float64); N = np.zeros(H, dtype=np.int64)
    with torch.no_grad():
        for batch in dl:
            x,u = batch[0].float(), batch[1].float()
            xhat,_ = model.pred_forward_test(x,u,False,args,-1)
            T = min(x.shape[1], xhat.shape[1], H)
            e = (x[:,:T,:]-xhat[:,:T,:]).cpu().numpy()[0]  # [T,D]
            SSE[:T] += (e**2).mean(axis=1)                 # mean over D
            N[:T]   += 1
    rmse_t = np.full(H, np.nan)
    valid = N>0
    rmse_t[valid] = np.sqrt(SSE[valid]/N[valid])
    return rmse_t

def maybe_load_loss(method):
    root = f"loss/{method}/{MODEL}"
    if not os.path.isdir(root): return None,None
    runs = sorted([d for d in os.listdir(root) if d.isdigit()], key=int)
    if not runs: return None,None
    last = runs[-1]
    tr = os.path.join(root,last,"loss_.txt")
    va = os.path.join(root,last,"loss_t.txt")
    return (np.loadtxt(tr) if os.path.exists(tr) else None,
            np.loadtxt(va) if os.path.exists(va) else None)

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # 1) 只生成一次环境和数据
    if args_cli.dataset == 'wastewater':
        base_args = dict(new_args.args, **new_args.ENV_PARAMS['wastewater'])
        base_args["env"] = MODEL
        train_set, val_set, test_set, draw_set = build_wastewater_datasets(
            steady_path=new_args.WASTEWATER_DATA['steady_state_path'],
            influent_paths=new_args.WASTEWATER_DATA['influent_paths'],
            profile_key=args_cli.influent_profile,
            seq_length=new_args.WASTEWATER_DATA['seq_length'],
            prediction_horizons=new_args.WASTEWATER_DATA['prediction_horizons'],
            train_frac=new_args.WASTEWATER_DATA['train_frac'],
            val_frac=new_args.WASTEWATER_DATA['val_frac'],
            expected_state_dim=new_args.WASTEWATER_DATA['expected_state_dim'],
            expected_influent_dim=new_args.WASTEWATER_DATA['expected_influent_dim'],
            normalize=new_args.WASTEWATER_DATA['normalize'],
        )
        test_draw = draw_set
        base_args["state_dim"] = new_args.WASTEWATER_DATA['expected_state_dim']
        base_args["act_dim"] = new_args.WASTEWATER_DATA['expected_influent_dim']
    else:
        from envs.cartpole import CartPoleEnv_adv as dreamer
        env = dreamer().unwrapped
        base_args = dict(new_args.args, **new_args.ENV_PARAMS[MODEL])
        base_args["state_dim"] = env.observation_space.shape[0]
        base_args["act_dim"]   = env.action_space.shape[0]
        base_args["env"] = MODEL
        RP = ReplayMemory(base_args, env, predict_evolution=True)
        test_draw = RP.dataset_test_draw

    # 2) 逐模型评估（共享同一 test_draw）
    RMSE = {}
    LOSSES = {}
    for method in METHODS:
        args = get_args(method)
        args["state_dim"] = base_args["state_dim"]
        args["act_dim"]   = base_args["act_dim"]
        model = load_model(method, args)
        curve = rmse_curve(model, test_draw, args, H=100)
        RMSE[method] = curve
        LOSSES[method] = maybe_load_loss(method)

    # 3) Log-loss 曲线（仅参考）
    plt.figure(figsize=(10,4))
    for method,color,ls in [("kovae","#1f77b4","-"), ("mamba","#d62728","-")]:
        tr, va = LOSSES[method]
        if va is not None: plt.plot(va, ls, label=f"{method} val", color=color)
        if tr is not None: plt.plot(tr, ls="--", alpha=0.5, label=f"{method} train", color=color)
    plt.yscale("log"); plt.xlabel("epoch"); plt.ylabel("loss (log)")
    plt.title("Validation/Train Loss (log scale)"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUTDIR}/loss_compare_log.png", dpi=160)

    # 4) RMSE@t（log-y）——主结论
    plt.figure(figsize=(10,4))
    t = np.arange(1, 101)
    for method,color in [("kovae","#1f77b4"), ("mamba","#d62728")]:
        plt.plot(t, RMSE[method], label=method, color=color)
    plt.yscale("log"); plt.xlabel("prediction step t"); plt.ylabel("RMSE@t (log)")
    plt.title("Per-step rollout RMSE (log scale)"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUTDIR}/rmse_curve_log.png", dpi=160)

    # 5) 关键步数 RMSE 表
    steps = [1,10,30,70]
    with open(f"{OUTDIR}/metrics.txt","w") as f:
        for s in steps:
            line = f"RMSE@{s}: " + "  ".join([f"{m}={RMSE[m][s-1]:.6f}" for m in METHODS])
            print(line); f.write(line+"\n")
    print("Saved:", f"{OUTDIR}/loss_compare_log.png", f"{OUTDIR}/rmse_curve_log.png", f"{OUTDIR}/metrics.txt")

if __name__ == "__main__":
    main()

