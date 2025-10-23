# eval_and_plot.py
import os, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from replay_fouling import ReplayMemory
import args_new as new_args

# ===== 配置区 =====
METHOD = "kovae"
MODEL  = "cartpole"
RUN_ID = 9
NUM_SAMPLES = 20
OUT_DIR = f"loss/{METHOD}/{MODEL}/{RUN_ID}/plots"

# ===== 加载 args =====
import args_new as new_args
args = dict(new_args.args, **new_args.ENV_PARAMS[MODEL])
args["env"] = MODEL
args["method"] = METHOD
args["control"] = False
args["continue_training"] = False

# kovae 的兜底
args.setdefault("z_dim", 16)
args.setdefault("h_dim", 64)
args.setdefault("alpha", 0.1)
args.setdefault("beta", 1e-3)
args.setdefault("eig_gamma", 0.0)
args.setdefault("grad_clip", 1.0)
args.setdefault("weight_decay", 1e-4)
args.setdefault("use_action", False)

# ===== 先设置保存/标准化文件的路径（关键）=====
fold_path = f"save_model/{METHOD}/{MODEL}"
os.makedirs(fold_path, exist_ok=True)
args["save_model_path"] = f"{fold_path}/model.pt"
args["save_opti_path"]  = f"{fold_path}/opti.pt"
args["shift_x"]         = f"{fold_path}/shift_x.txt"
args["scale_x"]         = f"{fold_path}/scale_x.txt"
args["shift_u"]         = f"{fold_path}/shift_u.txt"
args["scale_u"]         = f"{fold_path}/scale_u.txt"

# 输出目录
os.makedirs(OUT_DIR, exist_ok=True)

# ===== 环境 & 数据 =====
if MODEL == "cartpole":
    from envs.cartpole import CartPoleEnv_adv as dreamer
elif MODEL == "cartpole_V":
    from envs.cartpole_V import CartPoleEnv_adv as dreamer
else:
    raise ValueError("只示例了 cartpole/cartpole_V")

env = dreamer().unwrapped
args["state_dim"] = env.observation_space.shape[0]
args["act_dim"]   = env.action_space.shape[0]

from replay_fouling import ReplayMemory
replay_memory = ReplayMemory(args, env, predict_evolution=True)
test_draw = replay_memory.dataset_test_draw


# 数据（只用测试可视化那份）
replay_memory = ReplayMemory(args, env, predict_evolution=True)
test_draw = replay_memory.dataset_test_draw

# ===== 加载模型权重 =====
if METHOD == "kovae":
    from kovae_model import Koopman_Desko
elif METHOD == "mamba":
    from MamKO import Koopman_Desko
elif METHOD == "DKO":
    from DKO import Koopman_Desko
elif METHOD in ("MLP","LSTM","TRANS"):
    from MLP import Koopman_Desko
else:
    raise ValueError("未知方法")

model = Koopman_Desko(args)

# save_model 不分轮数，直接用固定路径
fold_path = f"save_model/{METHOD}/{MODEL}"
args["save_model_path"] = f"{fold_path}/model.pt"
args["save_opti_path"]  = f"{fold_path}/opti.pt"
args["shift_x"]         = f"{fold_path}/shift_x.txt"
args["scale_x"]         = f"{fold_path}/scale_x.txt"
args["shift_u"]         = f"{fold_path}/shift_u.txt"
args["scale_u"]         = f"{fold_path}/scale_u.txt"

model.parameter_restore(args)

#画若干条测试轨迹^_^
def to_np(t): return t.detach().cpu().numpy()
#循环开始！
import torch
import numpy as np

def to_torch_batch(x, u):
    """把 numpy 转成 torch.Tensor，并补 batch 维度"""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(u, np.ndarray):
        u = torch.from_numpy(u)
    # x: [T, Dx] → [1, T, Dx]
    if x.dim() == 2:
        x = x.unsqueeze(0)
    # u: [Du] or [T, Du] → [1, T, Du]
    if u.dim() == 1:
        T = x.size(1)
        u = u.unsqueeze(0).unsqueeze(0).expand(1, T, -1)
    elif u.dim() == 2:
        u = u.unsqueeze(0)
    return x.float(), u.float()

count = 0
for x, u in test_draw:
    # ✅ 转换 numpy→tensor，并补 batch 维
    x, u = to_torch_batch(x, u)

    # 前向预测
    xhat, aux = model.pred_forward_test(x, u, is_draw=False, args=args, epoch=-1)

    # 转回 numpy 以便画图
    xx   = x.squeeze(0).detach().cpu().numpy()      # [T, Dx]
    xh   = xhat.squeeze(0).detach().cpu().numpy()   # [T, Dx]
    T = min(len(xx), len(xh))

    # 画每个状态维度的轨迹对比
    fig, axs = plt.subplots(xx.shape[1], 1, figsize=(8, 2.2*xx.shape[1]), sharex=True)
    if xx.shape[1] == 1:
        axs = [axs]
    for i, ax in enumerate(axs):
        ax.plot(range(T), xx[:T, i],  label="gt", lw=2)
        ax.plot(range(T), xh[:T, i],  label="pred", lw=1.5)
        ax.set_ylabel(f"x[{i}]")
        ax.grid(True, alpha=0.3)
    axs[-1].set_xlabel("t")
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"traj_{count:03d}.png"), dpi=150)
    plt.close(fig)

    count += 1
    if count >= NUM_SAMPLES:
        break

print(f"轨迹图已保存到：{OUT_DIR}")

# ===== 可选：把 loss 曲线也画出来 =====
loss_dir = f"loss/{METHOD}/{MODEL}/{RUN_ID}"
loss_path = os.path.join(loss_dir, "loss_.txt")
vall_path = os.path.join(loss_dir, "loss_t.txt")
if os.path.exists(loss_path) and os.path.exists(vall_path):
    tr = np.loadtxt(loss_path)
    va = np.loadtxt(vall_path)
    plt.figure(figsize=(7,4))
    plt.plot(tr, label="train")
    plt.plot(va, label="val/test")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "loss_curve.png"), dpi=150)
    plt.close()
    print(f"Loss 曲线已保存到：{OUT_DIR}/loss_curve.png")
else:
    print("未找到 loss_.txt 或 loss_t.txt，跳过绘制 Loss 曲线。")
