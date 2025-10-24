# eval_and_plot.py
# 用来评估和可视化 KoVAE 模型在控制任务上的效果
# KoVAE 是用 VAE + Koopman 线性动力学来建模时间序列的
# 这里应该是把它应用到强化学习的环境预测上（cartpole 倒立摆）？
import os, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")    # 无界面模式，适合服务器跑
import matplotlib.pyplot as plt

from replay_fouling import ReplayMemory
import args_new as new_args

# 配置区
METHOD = "kovae"
MODEL  = "cartpole"    # 测试环境：倒立摆
RUN_ID = 9
NUM_SAMPLES = 20    # 画多少条轨迹
OUT_DIR = f"loss/{METHOD}/{MODEL}/{RUN_ID}/plots"

# 加载 args
# 从配置文件里读取基础参数，然后用特定环境的参数覆盖
import args_new as new_args
args = dict(new_args.args, **new_args.ENV_PARAMS[MODEL])
args["env"] = MODEL
args["method"] = METHOD
args["control"] = False    # 不做控制，只做预测评估
args["continue_training"] = False    # 不继续训练，只推理

# kovae 的兜底
# 这些是论文 Eq.(8) 损失函数里的权重和模型结构参数
args.setdefault("z_dim", 16)  # 潜在空间维度 k（论文里的 z_t 维度）
args.setdefault("h_dim", 64)  # GRU 隐层维度（论文用 GRU 作为编码器/解码器）
args.setdefault("alpha", 0.1)  # 预测损失权重（论文 Eq.8 的 α，匹配 z_t 和 bar{z}_t）
args.setdefault("beta", 1e-3)  # KL 散度权重（论文 Eq.8 的 β，正则化项）
args.setdefault("eig_gamma", 0.0)  # 特征值约束权重（论文 Eq.9 的物理约束，这里没用）
args.setdefault("grad_clip", 1.0)  # 梯度裁剪，防止训练不稳定
args.setdefault("weight_decay", 1e-4)  # L2 正则
args.setdefault("use_action", False)  # 是否把动作 u 也编码进去（这里好像不用）

# 先设置保存/标准化文件的路径（关键）
# 训练时应该保存了均值/方差用于标准化，推理时要用同样的参数
fold_path = f"save_model/{METHOD}/{MODEL}"
os.makedirs(fold_path, exist_ok=True)
args["save_model_path"] = f"{fold_path}/model.pt"  # 模型权重
args["save_opti_path"]  = f"{fold_path}/opti.pt"   # 优化器状态（这里用不到）
args["shift_x"]         = f"{fold_path}/shift_x.txt"  # 状态 x 的均值（标准化用）
args["scale_x"]         = f"{fold_path}/scale_x.txt"  # 状态 x 的标准差
args["shift_u"]         = f"{fold_path}/shift_u.txt"  # 动作 u 的均值
args["scale_u"]         = f"{fold_path}/scale_u.txt"  # 动作 u 的标准差

# 输出目录
os.makedirs(OUT_DIR, exist_ok=True)

# 环境 & 数据
# 根据论文，我们需要一个动力学系统来生成数据
# cartpole 是经典的倒立摆环境（x, x_dot, theta, theta_dot）
if MODEL == "cartpole":
    from envs.cartpole import CartPoleEnv_adv as dreamer
elif MODEL == "cartpole_V":
    from envs.cartpole_V import CartPoleEnv_adv as dreamer
else:
    raise ValueError("只示例了 cartpole/cartpole_V")

env = dreamer().unwrapped
args["state_dim"] = env.observation_space.shape[0]  # 状态维度（论文里的 d）
args["act_dim"]   = env.action_space.shape[0]       # 动作维度

# 加载测试数据
# ReplayMemory 应该是收集了一些轨迹数据（状态序列 x_1:T 和动作序列 u_1:T）
# 论文里的 x_t ∈ R^d 就对应这里的状态
from replay_fouling import ReplayMemory
replay_memory = ReplayMemory(args, env, predict_evolution=True)
test_draw = replay_memory.dataset_test_draw


# 数据（只用测试可视化那份）
replay_memory = ReplayMemory(args, env, predict_evolution=True)
test_draw = replay_memory.dataset_test_draw

# 加载模型权重
# 根据不同方法选择对应的模型实现
# kovae 就是论文里提出的 Koopman VAE
if METHOD == "kovae":
    from kovae_model import Koopman_Desko    # 应该实现了论文 Fig.1 的架构
elif METHOD == "mamba":
    from MamKO import Koopman_Desko    # 其他 baseline
elif METHOD == "DKO":
    from DKO import Koopman_Desko
elif METHOD in ("MLP","LSTM","TRANS"):
    from MLP import Koopman_Desko
else:
    raise ValueError("未知方法")

model = Koopman_Desko(args)

# 加载训练好的权重
# parameter_restore 应该会：
# 1. 加载模型参数（编码器、解码器、Koopman 矩阵 A）
# 2. 加载数据标准化参数（shift/scale）
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


# 可视化：画测试轨迹
# 目标：对比真实轨迹 x_1:T 和模型预测 \hat{x}_1:T
# 论文里说 KoVAE 能生成接近真实分布的时间序列
def to_torch_batch(x, u):
# 把 numpy 转成 torch.Tensor，并补 batch 维度。论文里的输入是 x_{t1:tN}（可能不规则），这里简化成规则采样
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(u, np.ndarray):
        u = torch.from_numpy(u)
    # x: [T, Dx] → [1, T, Dx]（加 batch 维）
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
    # 准备数据
    # x: 真实状态轨迹 [T, state_dim]
    # u: 动作序列 [T, act_dim] 或单个动作
    x, u = to_torch_batch(x, u)

    # 前向预测
    # 论文 Sec 4.1-4.2：
    # 1. 编码器 q(z_t | z_<t, x_≤t) 把 x 编码到潜在空间
    # 2. Koopman 线性映射 z_t = A * z_{t-1}（论文 Eq.6）
    # 3. 解码器 p(x_t | z_t) 重建状态
    xhat, aux = model.pred_forward_test(x, u, is_draw=False, args=args, epoch=-1)

    # 转回 numpy 以便画图
    xx   = x.squeeze(0).detach().cpu().numpy()      # [T, Dx]
    xh   = xhat.squeeze(0).detach().cpu().numpy()   # [T, Dx]
    T = min(len(xx), len(xh))

    # 画每个状态维度的对比图
    # 例如 cartpole 有 4 维：[位置, 速度, 角度, 角速度]
    fig, axs = plt.subplots(xx.shape[1], 1, figsize=(8, 2.2*xx.shape[1]), sharex=True)
    if xx.shape[1] == 1:
        axs = [axs]
    for i, ax in enumerate(axs):
        ax.plot(range(T), xx[:T, i],  label="gt", lw=2)
        ax.plot(range(T), xh[:T, i],  label="pred", lw=1.5)
        ax.set_ylabel(f"x[{i}]")    # 状态第 i 维
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

# 把 loss 曲线也画出来
# 对应论文 Eq.(8) 的总损失 L（重构 + 预测 + KL 散度）
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
    print(f"^_^Loss 曲线已保存到：{OUT_DIR}/loss_curve.png")
else:
    print("( ･᷄ὢ･᷅ )未找到 loss_.txt 或 loss_t.txt，跳过绘制 Loss 曲线。")

