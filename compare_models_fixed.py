#!/usr/bin/env python3
"""
修复后的模型对比脚本 - 使用真正的多步预测
Compare MamKO vs KoVAE with proper multi-step prediction
"""

import numpy as np
import torch
import os
from matplotlib import pyplot as plt

# 加载数据标准化统计量
shift_x = np.loadtxt('save_model/mamba/waste_water/shift_x.txt')
scale_x = np.loadtxt('save_model/mamba/waste_water/scale_x.txt')
shift_u = np.loadtxt('save_model/mamba/waste_water/shift_u.txt')
scale_u = np.loadtxt('save_model/mamba/waste_water/scale_u.txt')

# 防止除以零
scale_x[np.where(scale_x == 0)] = 1
scale_u[np.where(scale_u == 0)] = 1

# 找出关键维度 (标准差最大的 5 个)
top_dims = np.argsort(scale_x)[-5:]
print(f"关键维度 (标准差最大): {top_dims}")
print(f"对应标准差: {scale_x[top_dims]}")
print()

# =========================
# 加载 MamKO
# =========================
print("=== 加载 MamKO ===")
import args_new as new_args
args_mamko = dict(new_args.args, **new_args.ENV_PARAMS['waste_water'])
args_mamko['method'] = 'mamba'
args_mamko['env'] = 'waste_water'
args_mamko['continue_training'] = False

from waste_water_system import waste_water_system as dreamer
env = dreamer()
env = env.unwrapped
args_mamko['state_dim'] = env.observation_space.shape[0]
args_mamko['act_dim'] = env.action_space.shape[0]
args_mamko['control'] = False

fold_path = 'save_model/mamba/waste_water'
args_mamko['save_model_path'] = fold_path + '/model.pt'
args_mamko['save_opti_path'] = fold_path + '/opti.pt'
args_mamko['shift_x'] = fold_path + '/shift_x.txt'
args_mamko['scale_x'] = fold_path + '/scale_x.txt'
args_mamko['shift_u'] = fold_path + '/shift_u.txt'
args_mamko['scale_u'] = fold_path + '/scale_u.txt'

from MamKO import Koopman_Desko as MamKO_Model
mamko = MamKO_Model(args_mamko)
mamko.parameter_restore(args_mamko)
print("MamKO 加载成功\n")

# =========================
# 加载 KoVAE
# =========================
print("=== 加载 KoVAE ===")
args_kovae = dict(new_args.args, **new_args.ENV_PARAMS['waste_water'])
args_kovae['method'] = 'kovae'
args_kovae['env'] = 'waste_water'
args_kovae['state_dim'] = env.observation_space.shape[0]
args_kovae['act_dim'] = env.action_space.shape[0]
args_kovae['continue_training'] = False
args_kovae.setdefault('z_dim', 16)
args_kovae.setdefault('h_dim', 64)
args_kovae.setdefault('alpha', 0.1)
args_kovae.setdefault('beta', 1e-3)
args_kovae.setdefault('gamma', 0.0)
args_kovae.setdefault('grad_clip', 1.0)
args_kovae.setdefault('weight_decay', 1e-4)
args_kovae.setdefault('use_action', False)

fold_path_kovae = 'save_model/kovae/waste_water'
args_kovae['save_model_path'] = fold_path_kovae + '/model.pt'
args_kovae['save_opti_path'] = fold_path_kovae + '/opti.pt'
args_kovae['shift_x'] = fold_path_kovae + '/shift_x.txt'
args_kovae['scale_x'] = fold_path_kovae + '/scale_x.txt'
args_kovae['shift_u'] = fold_path_kovae + '/shift_u.txt'
args_kovae['scale_u'] = fold_path_kovae + '/scale_u.txt'

# Check if KoVAE checkpoint exists
if not os.path.exists(args_kovae['save_model_path']):
    print(f"错误: KoVAE 模型不存在 at {args_kovae['save_model_path']}")
    exit(1)

from kovae_model import Koopman_Desko as KoVAE_Model
kovae = KoVAE_Model(args_kovae)
kovae.parameter_restore(args_kovae)
print("KoVAE 加载成功 (使用修复后的预测函数)\n")

# =========================
# 加载测试数据
# =========================
print("=== 加载测试数据 ===")
test_draw = torch.load(new_args.args['SAVE_DRAW'])
print(f"测试数据集大小: {len(test_draw)} 样本")

# 取一个测试样本
x_test, u_test = test_draw[0]
x_test = x_test.unsqueeze(0)  # [1, T, state_dim]
u_test = u_test.unsqueeze(0)  # [1, T-1, act_dim]
print(f"测试样本形状: x={x_test.shape}, u={u_test.shape}\n")

# =========================
# 生成预测 (使用修复后的 KoVAE)
# =========================
print("=== 生成预测 ===")
mamko_pred, mamko_aux = mamko.pred_forward_test(x_test, u_test, False, args_mamko, 0)
kovae_pred, kovae_aux = kovae.pred_forward_test(x_test, u_test, False, args_kovae, 0)

print(f"MamKO 预测形状: {mamko_pred.shape}")
print(f"KoVAE 预测形状: {kovae_pred.shape}")
print()

# =========================
# 反标准化
# =========================
def denormalize(x_norm, shift, scale):
    return x_norm * scale + shift

x_test_np = x_test.numpy()
mamko_pred_np = mamko_pred.numpy()
kovae_pred_np = kovae_pred.numpy()

# 截取预测部分 (old_horizon 之后)
old_horizon = args_mamko['old_horizon']
pred_horizon = args_mamko['pred_horizon']

# MamKO 的输出是 [time_steps, batch, state_dim]，需要转置
if mamko_pred_np.shape[0] != 1:  # 如果第一维不是 batch
    mamko_pred_np = np.transpose(mamko_pred_np, (1, 0, 2))

truth_norm = x_test_np[0, old_horizon:old_horizon+pred_horizon, :]
mamko_norm = mamko_pred_np[0, :pred_horizon, :]
kovae_norm = kovae_pred_np[0, old_horizon:old_horizon+pred_horizon, :]

truth_denorm = denormalize(truth_norm, shift_x, scale_x)
mamko_denorm = denormalize(mamko_norm, shift_x, scale_x)
kovae_denorm = denormalize(kovae_norm, shift_x, scale_x)

# =========================
# 计算误差 (只在关键维度上)
# =========================
print("=" * 70)
print("关键维度预测误差对比 (反标准化后)")
print("=" * 70)

mamko_rmses = []
kovae_rmses = []

for i, dim in enumerate(top_dims):
    truth_vals = truth_denorm[:, dim]
    mamko_vals = mamko_denorm[:, dim]
    kovae_vals = kovae_denorm[:, dim]

    mamko_rmse = np.sqrt(np.mean((mamko_vals - truth_vals)**2))
    kovae_rmse = np.sqrt(np.mean((kovae_vals - truth_vals)**2))

    mean_val = np.mean(np.abs(truth_vals))
    mamko_rel = (mamko_rmse / mean_val * 100) if mean_val > 0 else 0
    kovae_rel = (kovae_rmse / mean_val * 100) if mean_val > 0 else 0

    mamko_rmses.append(mamko_rel)
    kovae_rmses.append(kovae_rel)

    print(f"维度 {dim:3d}:")
    print(f"  MamKO - RMSE: {mamko_rmse:8.2f}, 相对RMSE: {mamko_rel:6.2f}%")
    print(f"  KoVAE - RMSE: {kovae_rmse:8.2f}, 相对RMSE: {kovae_rel:6.2f}%")
    ratio = kovae_rel / mamko_rel if mamko_rel > 0 else float('inf')
    winner = "MamKO ✓" if mamko_rel < kovae_rel else "KoVAE ✓"
    print(f"  KoVAE/MamKO 比值: {ratio:.2f}x  【{winner}】")
    print()

print("=" * 70)
print("平均相对误差:")
print(f"  MamKO: {np.mean(mamko_rmses):.2f}%")
print(f"  KoVAE: {np.mean(kovae_rmses):.2f}%")
print("=" * 70)
print()

# =========================
# 可视化对比
# =========================
print("生成可视化对比图...")
fig, axes = plt.subplots(5, 1, figsize=(12, 10))
fig.suptitle('MamKO vs KoVAE 关键维度预测对比 (修复后)', fontsize=14, fontweight='bold')

time_steps = np.arange(len(truth_denorm))

for i, (ax, dim) in enumerate(zip(axes, top_dims)):
    ax.plot(time_steps, truth_denorm[:, dim], 'k-', linewidth=2, label='真实值')
    ax.plot(time_steps, mamko_denorm[:, dim], 'b--', linewidth=1.5, alpha=0.8, label='MamKO')
    ax.plot(time_steps, kovae_denorm[:, dim], 'r:', linewidth=1.5, alpha=0.8, label='KoVAE (修复)')

    ax.set_ylabel(f'维度 {dim}', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    # 添加误差信息
    mamko_err = mamko_rmses[i]
    kovae_err = kovae_rmses[i]
    ax.text(0.02, 0.98, f'MamKO: {mamko_err:.1f}% | KoVAE: {kovae_err:.1f}%',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

axes[-1].set_xlabel('时间步', fontsize=11)
plt.tight_layout()
plt.savefig('comparison_mamko_vs_kovae_fixed.png', dpi=150, bbox_inches='tight')
print("对比图已保存: comparison_mamko_vs_kovae_fixed.png")
print()

print("=" * 70)
print("【结论】")
print("=" * 70)
print("现在使用了修复后的 KoVAE 预测函数:")
print("  - 使用 Koopman 算子 A 做线性递推")
print("  - 每步解码得到预测状态")
print("  - 与真实未来状态对比")
print()
if np.mean(kovae_rmses) < np.mean(mamko_rmses):
    print("KoVAE 的真实预测性能优于 MamKO！")
elif np.mean(kovae_rmses) > 2 * np.mean(mamko_rmses):
    print("KoVAE 的真实预测性能仍然较差，可能:")
    print("  1. Koopman 算子 A 没有学到正确的动力学")
    print("  2. 训练损失主要优化了重构，而不是预测")
    print("  3. 需要调整训练策略或超参数")
else:
    print("两个模型的预测性能相近。")
print("=" * 70)
