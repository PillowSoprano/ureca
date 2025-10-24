# compare_loss.py
# PLEASE DON'T FAIL!!!!!!!!!
# 这个脚本用来对比不同方法在同一任务上的性能
# 读完论文后，我想看看 KoVAE 相比其他 baseline（Mamba, DKO, MLP）到底好在哪
# 论文 Table 1-3 就是这样做定量对比的
import os, glob, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 配置
MODEL   = "cartpole"                        # 测试环境（倒立摆）
METHODS = ["kovae", "mamba", "DKO", "MLP"]  # 要对比的方法
# kovae: 论文提出的方法（VAE + Koopman 线性先验）
# mamba/DKO/MLP: 其他 baseline（可能是其他序列建模方法）
RUN_IDS = list(range(10))  # 跑了 10 个不同随机种子（论文里也是多 seed 取均值±std）
SMOOTH  = 0.9              # EMA 平滑系数，让曲线更平滑（0=不平滑）
OUTDIR  = f"loss/compare/{MODEL}"
os.makedirs(OUTDIR, exist_ok=True)

def load_runs(method, metric="val"):
# 返回：list[np.ndarray]，每个元素是一条曲线（一个 run）
    runs = []
    for i in RUN_IDS:
        base = f"loss/{method}/{MODEL}/{i}"
        # loss_t.txt: 验证/测试损失（对应论文的 discriminative/predictive score）
        # loss_.txt:  训练损失（用来看是否过拟合）
        f = os.path.join(base, "loss_t.txt" if metric=="val" else "loss_.txt")
        if not os.path.exists(f): 
            continue
        y = np.loadtxt(f).astype(float).flatten()
        runs.append(y)
    return runs

def align_and_stack(curves):
# 不同 run 可能训练了不同的 epoch 数，对齐到最短长度
# 返回：[N_runs, L] 的矩阵
    L = min(len(c) for c in curves)
    X = np.stack([c[:L] for c in curves], axis=0)  # [N, L]
    return X

def ema(y, alpha=0.9):
# 指数移动平均（EMA）平滑曲线
# 让训练曲线看起来不那么抖动
# z[t] = α * z[t-1] + (1-α) * y[t]
    if alpha<=0: return y
    z = np.zeros_like(y)
    z[0]=y[0]
    for t in range(1,len(y)):
        z[t] = alpha*z[t-1] + (1-alpha)*y[t]
    return z

def metrics(train_mat, val_mat):
# 输入形状 [N_runs, L]；输出字典（跨 seed 取均值±std）
# 论文里的评估指标：
# - Best val: 验证集上的最低损失（越低越好）
# - Best epoch: 达到最低损失的轮数（越早越好，说明收敛快）
# - Final val: 最后几轮的平均损失（看是否稳定）
# - AUC: 整个训练过程的损失积分（越低越好，说明全程都不错）
# - Gap: train-val 差距（>0 说明过拟合）
    def _one(mat):
        L = mat.shape[1]
        best_idx = np.argmin(mat, axis=1)              # [N]
        best_val = mat[np.arange(len(best_idx)), best_idx]
        final_val = mat[:, max(0, L-10):].mean(axis=1)
        # 损失曲线下面积（近似积分，步长=1）
        auc = mat.mean(axis=1) * L                     # 近似积分（步长=1）
        return best_val, best_idx, final_val, auc

    bval, bepoch, fval, auc = _one(val_mat)
    bval_mu, bval_sd = bval.mean(), bval.std()
    bepoch_mu, bepoch_sd = bepoch.mean(), bepoch.std()
    fval_mu, fval_sd = fval.mean(), fval.std()
    auc_mu, auc_sd = auc.mean(), auc.std()

    # 过拟合程度：final(train) - final(val)
    # 如果 train loss 远低于 val loss，说明过拟合了
    # 论文说 VAE 的正则化项（KL 散度）可以缓解过拟合
    if train_mat is not None and train_mat.shape == val_mat.shape:
        gap = train_mat[:, -10:].mean(axis=1) - val_mat[:, -10:].mean(axis=1)
        gap_mu, gap_sd = gap.mean(), gap.std()
    else:
        gap_mu = gap_sd = np.nan

    return dict(
        best_val=(bval_mu, bval_sd),       # 最佳验证损失
        best_epoch=(bepoch_mu, bepoch_sd), # 最佳轮数
        final_val=(fval_mu, fval_sd),      # 最终稳定损失
        auc=(auc_mu, auc_sd),              # 曲线下面积
        gap_final=(gap_mu, gap_sd),        # 过拟合差距
    )

summary = {}  # 保存每个方法的统计结果

# 画图并统计
plt.figure(figsize=(9,5))
for m in METHODS:
    # 加载该方法在所有 seed 下的验证损失
    val_runs = load_runs(m, "val")
    if len(val_runs) == 0:
        print(f"[warn] no runs found for {m}")
        continue
    
    # 平滑并对齐到相同长度
    val_mat = align_and_stack([ema(r, SMOOTH) for r in val_runs])

   # 加载训练损失（用于计算 train-val gap）
    tr_runs = load_runs(m, "train")
    tr_mat = align_and_stack([ema(r, SMOOTH) for r in tr_runs]) if len(tr_runs) == len(val_runs) else None

    # 计算均值和标准差
    L = val_mat.shape[1]
    mu = val_mat.mean(axis=0)  # [L] 每个 epoch 的平均损失
    sd = val_mat.std(axis=0)   # [L] 每个 epoch 的标准差

    # 画均值曲线，用阴影表示±std（论文 Table 就是这样报告的）
    x = np.arange(L)
    plt.plot(x, mu, label=m)
    plt.fill_between(x, mu - sd, mu + sd, alpha=0.15)

    # 计算各项指标
    summary[m] = metrics(tr_mat, val_mat)

plt.xlabel("epoch"); plt.ylabel("val loss")
plt.title(f"Validation Loss (mean±std over seeds) - {MODEL}")
plt.grid(alpha=0.3); plt.legend()
plt.tight_layout()
fig_path = os.path.join(OUTDIR, "val_loss_compare.png")
plt.savefig(fig_path, dpi=180)
plt.close()
print(f"曲线图已保存：{fig_path}")

# 打印表格并另存为 csv
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

