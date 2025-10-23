# modeling_compare_log.py
import os, numpy as np, torch, matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import args_new as new_args
from replay_fouling import ReplayMemory

plt.rcParams["figure.dpi"] = 160

MODEL   = "cartpole"                 # 如需换环境在这里改
METHODS = ["mamba", "kovae"]         # MamKO vs KoVAE
BATCH   = 128
VAL_RATIO = 0.2

def make_env():
    from envs.cartpole import CartPoleEnv_adv as dreamer
    return dreamer().unwrapped

def build_args(method: str, env):
    args = dict(new_args.args)
    args.update(new_args.ENV_PARAMS[MODEL])

    args["env"]     = MODEL
    args["method"]  = method
    args["control"] = False

    # 先补齐维度（非常关键，MamKO 在 __init__ 里会用到）
    args["state_dim"] = int(env.observation_space.shape[0])
    args["act_dim"]   = int(env.action_space.shape[0])
    args.setdefault("obs_dim", args["state_dim"])

    # 路径兜底
    fold = f"save_model/{method}/{MODEL}"
    os.makedirs(fold, exist_ok=True)
    args.setdefault("save_model_path", f"{fold}/model.pt")
    args.setdefault("save_opti_path",  f"{fold}/opti.pt")
    args.setdefault("shift_x",         f"{fold}/shift_x.txt")
    args.setdefault("scale_x",         f"{fold}/scale_x.txt")
    args.setdefault("shift_u",         f"{fold}/shift_u.txt")
    args.setdefault("scale_u",         f"{fold}/scale_u.txt")
    return args

def load_model(method: str, args):
    if method == "kovae":
        from kovae_model import Koopman_Desko
    elif method == "mamba":
        from MamKO import Koopman_Desko
    else:
        raise ValueError(f"Unknown method {method}")
    m = Koopman_Desko(args)

    # 恢复权重
    if hasattr(m, "parameter_restore"): m.parameter_restore(args)
    elif hasattr(m, "restore"):         m.restore()

    # 设备：优先已有 device，其次 mps->cuda->cpu
    if hasattr(m, "device"):
        device = m.device
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    m.device = device

    # MamKO 只有 .net，做个别名方便兜底
    if not hasattr(m, "model") and hasattr(m, "net"):
        m.model = m.net
    try:
        if hasattr(m, "model"):
            m.model.to(device)
    except Exception:
        pass
    return m

def _get_split_dataset(mem: ReplayMemory, split: str):
    if split == "train":
        return mem.dataset_train
    if split == "test":
        return mem.dataset_test
    if split == "val":
        if hasattr(mem, "dataset_val"):
            return mem.dataset_val
        # 从 train 切出 val
        full = mem.dataset_train
        n = len(full)
        n_val = max(1, int(round(VAL_RATIO * n)))
        n_train = max(1, n - n_val)
        g = torch.Generator().manual_seed(0)
        val_ds, _ = random_split(full, [n_val, n_train], generator=g)
        return val_ds
    raise ValueError(split)

@torch.no_grad()
def one_step_mse(model, args, env, split="val", batch_size=BATCH):
    """
    使用序列数据 (x_seq, u_seq) 评估 L 步预测的 MSE（对齐到未来窗口）。
    - x_seq: [B, O+L, Dx]
    - u_seq: [B, O+L-1, Du]
    预测与真值都用同一归一化空间比较，便于方法间公平对比。
    """
    O = int(args["old_horizon"])
    L = int(args["pred_horizon"])

    # 关键：predict_evolution=True 才会返回 (x_seq, u_seq)
    mem = ReplayMemory(args, env, predict_evolution=True)
    ds  = _get_split_dataset(mem, split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    dev = model.device
    mse_sum = 0.0
    n_samp  = 0

    for x_seq, u_seq in loader:
        # x_seq: [B, O+L, Dx], u_seq: [B, O+L-1, Du]
        x_seq = x_seq.float().to(dev)
        u_seq = u_seq.float().to(dev)

        # 未来真值窗口：[B, L, Dx]
        y_true = x_seq[:, O:O+L, :].detach().cpu().numpy()

        # 分方法获取预测序列
        y_pred = None
        if hasattr(model, "net"):  # MamKO
            loss, y_pred_seq = model.net(x_seq, u_seq)   # loss 标量, y_pred_seq: [L, B, Dx]（见 selective_scan）
            y_pred = np.transpose(y_pred_seq, (1, 0, 2)) # -> [B, L, Dx]
        else:
            # KoVAE 壳：拼接 xin，再过 self.model 得 xhat（长度与 x_seq 相同）
            if hasattr(model, "_make_batch"):
                x, u, xin = model._make_batch((x_seq, u_seq))
            else:
                # 兜底：只用 x
                xin = x_seq
            # 取出 αβγ；若没有就全 0
            alpha = float(getattr(model, "alpha", 0.0))
            beta  = float(getattr(model, "beta",  0.0))
            gamma = float(getattr(model, "gamma", 0.0))
            loss, xhat, _ = model.model(xin, alpha, beta, gamma)  # xhat: [B, O+L, Dx]
            y_pred = xhat[:, O:O+L, :].detach().cpu().numpy()

        if y_pred is None:
            raise RuntimeError("Failed to obtain prediction sequence from the model.")

        # [B, L, Dx] → per-sample MSE
        err = ((y_pred - y_true) ** 2).reshape(y_true.shape[0], -1).mean(axis=1)
        mse_sum += float(err.sum())
        n_samp  += int(err.shape[0])

    return mse_sum / max(1, n_samp)

def plot_log_bar(method2val, title, outpng):
    methods = list(method2val.keys())
    vals = [method2val[m] for m in methods]
    plt.figure()
    plt.yscale("log")
    plt.bar(methods, vals)
    plt.ylabel("MSE (log scale)")
    plt.title(title)
    for i, v in enumerate(vals):
        if np.isfinite(v):
            plt.text(i, v, f"{v:.3e}", ha="center", va="bottom", rotation=90, fontsize=8)
    os.makedirs(os.path.dirname(outpng), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpng)
    plt.close()

def main():
    outdir = f"results/{MODEL}/compare"
    os.makedirs(outdir, exist_ok=True)

    env = make_env()

    for split in ["train", "val", "test"]:
        summary = {}
        for m in METHODS:
            args   = build_args(m, env)   # 先用 env 补好维度
            model  = load_model(m, args)  # 再实例化（MamKO 需要）
            summary[m] = one_step_mse(model, args, env, split=split)
        plot_log_bar(summary, f"Modeling {split} MSE", f"{outdir}/modeling_{split}_log.png")
        print(split, {k: f"{v:.3e}" for k, v in summary.items()})

if __name__ == "__main__":
    main()
