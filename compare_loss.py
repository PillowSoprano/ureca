# compare_loss.py
import os, glob, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== 配置 =====
MODEL   = "cartpole"                        # or "cartpole_V"
METHODS = ["kovae", "mamba", "DKO", "MLP"]  # 你想比较的若干方法
RUN_IDS = list(range(10))                   # 你一共跑了 10 轮（0..9）
SMOOTH  = 0.9                               # EMA 平滑系数，0=不平滑
OUTDIR  = f"loss/compare/{MODEL}"
os.makedirs(OUTDIR, exist_ok=True)

def load_runs(method, metric="val"):
    """返回：list[np.ndarray]，每个元素是一条曲线（一个 run）"""
    runs = []
    for i in RUN_IDS:
        base = f"loss/{method}/{MODEL}/{i}"
        f = os.path.join(base, "loss_t.txt" if metric=="val" else "loss_.txt")
        if not os.path.exists(f): 
            continue
        y = np.loadtxt(f).astype(float).flatten()
        runs.append(y)
    return runs

def align_and_stack(curves):
    """不同长度对齐到最短长度"""
    L = min(len(c) for c in curves)
    X = np.stack([c[:L] for c in curves], axis=0)  # [N, L]
    return X

def ema(y, alpha=0.9):
    if alpha<=0: return y
    z = np.zeros_like(y)
    z[0]=y[0]
    for t in range(1,len(y)):
        z[t] = alpha*z[t-1] + (1-alpha)*y[t]
    return z

def metrics(train_mat, val_mat):
    """输入形状 [N_runs, L]；输出字典（跨 seed 取均值±std）"""
    def _one(mat):
        L = mat.shape[1]
        best_idx = np.argmin(mat, axis=1)              # [N]
        best_val = mat[np.arange(len(best_idx)), best_idx]
        final_val = mat[:, max(0, L-10):].mean(axis=1)
        auc = mat.mean(axis=1) * L                     # 近似积分（步长=1）
        return best_val, best_idx, final_val, auc

    bval, bepoch, fval, auc = _one(val_mat)
    bval_mu, bval_sd = bval.mean(), bval.std()
    bepoch_mu, bepoch_sd = bepoch.mean(), bepoch.std()
    fval_mu, fval_sd = fval.mean(), fval.std()
    auc_mu, auc_sd = auc.mean(), auc.std()

    # 过拟合程度：final(train) - final(val)（>0 说明 val 更好；<0 说明过拟合）
    if train_mat is not None and train_mat.shape == val_mat.shape:
        gap = train_mat[:, -10:].mean(axis=1) - val_mat[:, -10:].mean(axis=1)
        gap_mu, gap_sd = gap.mean(), gap.std()
    else:
        gap_mu = gap_sd = np.nan

    return dict(
        best_val=(bval_mu, bval_sd),
        best_epoch=(bepoch_mu, bepoch_sd),
        final_val=(fval_mu, fval_sd),
        auc=(auc_mu, auc_sd),
        gap_final=(gap_mu, gap_sd),
    )

summary = {}

# ===== 画图并统计 =====
plt.figure(figsize=(9,5))
for m in METHODS:
    val_runs = load_runs(m, "val")
    if len(val_runs)==0:
        print(f"[warn] no runs found for {m}")
        continue
    val_mat = align_and_stack([ema(r, SMOOTH) for r in val_runs])

    # 训练集（可选）用于 gap 统计
    tr_runs = load_runs(m, "train")
    tr_mat = align_and_stack([ema(r, SMOOTH) for r in tr_runs]) if len(tr_runs)==len(val_runs) else None

    L = val_mat.shape[1]
    mu = val_mat.mean(axis=0)
    sd = val_mat.std(axis=0)

    x = np.arange(L)
    plt.plot(x, mu, label=m)
    plt.fill_between(x, mu-sd, mu+sd, alpha=0.15)

    summary[m] = metrics(tr_mat, val_mat)

plt.xlabel("epoch"); plt.ylabel("val loss")
plt.title(f"Validation Loss (mean±std over seeds) - {MODEL}")
plt.grid(alpha=0.3); plt.legend()
plt.tight_layout()
fig_path = os.path.join(OUTDIR, "val_loss_compare.png")
plt.savefig(fig_path, dpi=180)
plt.close()
print(f"曲线图已保存：{fig_path}")

# ===== 打印表格并另存为 csv =====
import csv
csv_path = os.path.join(OUTDIR, "loss_summary.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method","best_val(μ±σ)","best_epoch(μ±σ)","final_val(μ±σ)","AUC(μ±σ)","gap_final(μ±σ)"])
    for m in METHODS:
        if m not in summary: continue
        s = summary[m]
        fmt = lambda x: f"{x[0]:.4g}±{x[1]:.4g}" if all(np.isfinite(x)) else "na"
        w.writerow([
            m,
            fmt(s["best_val"]),
            fmt(s["best_epoch"]),
            fmt(s["final_val"]),
            fmt(s["auc"]),
            fmt(s["gap_final"]),
        ])
print(f"指标汇总已保存：{csv_path}")

# 同时在终端打印
from pprint import pprint
print("\n==== Summary (μ±σ) ====")
pprint(summary)
