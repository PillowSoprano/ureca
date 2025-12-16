#!/usr/bin/env python3
"""
测试 KoVAE 修复后的预测功能
在 Codespace 中运行此脚本来验证修复是否有效
"""

import numpy as np
import torch
import os

print("=" * 70)
print("测试 KoVAE 修复后的预测功能")
print("=" * 70)
print()

# 检查模型是否存在
kovae_model_path = 'save_model/kovae/waste_water/model.pt'
if not os.path.exists(kovae_model_path):
    print(f"错误: 找不到 KoVAE 模型 at {kovae_model_path}")
    print("请确保已经训练过 KoVAE 模型")
    exit(1)

print("✓ KoVAE 模型存在")
print()

# 加载配置
import args_new as new_args
args = dict(new_args.args, **new_args.ENV_PARAMS['waste_water'])
args['method'] = 'kovae'
args['env'] = 'waste_water'

from waste_water_system import waste_water_system as dreamer
env = dreamer()
env = env.unwrapped
args['state_dim'] = env.observation_space.shape[0]
args['act_dim'] = env.action_space.shape[0]
args['continue_training'] = False

# KoVAE 参数
args.setdefault('z_dim', 16)
args.setdefault('h_dim', 64)
args.setdefault('alpha', 0.1)
args.setdefault('beta', 1e-3)
args.setdefault('gamma', 0.0)
args.setdefault('grad_clip', 1.0)
args.setdefault('weight_decay', 1e-4)
args.setdefault('use_action', False)

fold_path = 'save_model/kovae/waste_water'
args['save_model_path'] = fold_path + '/model.pt'
args['save_opti_path'] = fold_path + '/opti.pt'
args['shift_x'] = fold_path + '/shift_x.txt'
args['scale_x'] = fold_path + '/scale_x.txt'
args['shift_u'] = fold_path + '/shift_u.txt'
args['scale_u'] = fold_path + '/scale_u.txt'

# 加载模型
print("加载 KoVAE 模型...")
from kovae_model import Koopman_Desko
model = Koopman_Desko(args)
model.parameter_restore(args)
print("✓ 模型加载成功")
print()

# 加载测试数据
print("加载测试数据...")
import os
draw_path = new_args.args['SAVE_DRAW']

# 如果测试数据文件不存在，生成一个
if not os.path.exists(draw_path):
    print("  测试数据文件不存在，从 replay memory 生成...")
    from replay_fouling import ReplayMemory
    replay_memory = ReplayMemory(args, env, predict_evolution=True)
    test_draw = replay_memory.dataset_test_draw
    print(f"  ✓ 从 replay memory 生成了 {len(test_draw)} 个测试样本")
else:
    test_draw = torch.load(draw_path)
    print(f"  ✓ 从文件加载了 {len(test_draw)} 个测试样本")

x_test, u_test = test_draw[0]

# 转换成 torch tensor (如果是 numpy)
if isinstance(x_test, np.ndarray):
    x_test = torch.from_numpy(x_test).float()
if isinstance(u_test, np.ndarray):
    u_test = torch.from_numpy(u_test).float()

# 添加 batch 维度
if x_test.dim() == 2:
    x_test = x_test.unsqueeze(0)
if u_test.dim() == 2:
    u_test = u_test.unsqueeze(0)

print(f"✓ 测试数据形状: x={x_test.shape}, u={u_test.shape}")
print()

# 运行修复后的预测
print("运行修复后的预测函数...")
print("  - 使用 Koopman 算子 A 做线性递推")
print("  - 每步解码得到预测状态")
print()

pred, aux = model.pred_forward_test(x_test, u_test, False, args, 0)
print(f"✓ 预测完成!")
print(f"  预测输出形状: {pred.shape}")
print(f"  预测损失: {aux['pred_loss']:.6f}")
print()

# 加载标准化统计量
shift_x = np.loadtxt(args['shift_x'])
scale_x = np.loadtxt(args['scale_x'])
scale_x[np.where(scale_x == 0)] = 1

# 计算关键维度的误差
top_dims = np.argsort(scale_x)[-5:]
print("=" * 70)
print("关键维度预测误差 (反标准化后)")
print("=" * 70)

old_horizon = args['old_horizon']
pred_horizon = args['pred_horizon']

x_test_np = x_test.numpy()
pred_np = pred.numpy()

truth_norm = x_test_np[0, old_horizon:old_horizon+pred_horizon, :]
pred_norm = pred_np[0, old_horizon:old_horizon+pred_horizon, :]

truth_denorm = truth_norm * scale_x + shift_x
pred_denorm = pred_norm * scale_x + shift_x

rel_errors = []
for dim in top_dims:
    truth_vals = truth_denorm[:, dim]
    pred_vals = pred_denorm[:, dim]

    rmse = np.sqrt(np.mean((pred_vals - truth_vals)**2))
    mean_val = np.mean(np.abs(truth_vals))
    rel_rmse = (rmse / mean_val * 100) if mean_val > 0 else 0

    rel_errors.append(rel_rmse)
    print(f"维度 {dim:3d}: RMSE={rmse:8.2f}, 相对RMSE={rel_rmse:6.2f}%")

print()
print(f"平均相对误差: {np.mean(rel_errors):.2f}%")
print("=" * 70)
print()

# 对比之前的结果
print("【结果对比】")
print()
print("之前 (使用重构，不是预测):")
print("  KoVAE: 120-143% 相对误差 ❌")
print()
print(f"现在 (使用 Koopman 算子预测):")
print(f"  KoVAE: {np.mean(rel_errors):.1f}% 相对误差")
print()
print("MamKO (参考):")
print("  MamKO: 14-15% 相对误差 ✓")
print()

if np.mean(rel_errors) < 30:
    print("✓ 修复成功！KoVAE 现在使用真正的预测了")
    print("  预测性能提升显著")
elif np.mean(rel_errors) < 100:
    print("⚠ 部分改善，但 KoVAE 预测仍不如 MamKO")
    print("  可能需要调整训练策略")
else:
    print("❌ KoVAE 预测性能仍然很差")
    print("  Koopman 算子可能没有学到正确的动力学")
print()

print("=" * 70)
print("如需完整对比，运行: python compare_models_fixed.py")
print("=" * 70)
