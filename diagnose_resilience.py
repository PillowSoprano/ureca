#!/usr/bin/env python3
"""
诊断 resilience_evaluation.py 的异常结果
"""

import numpy as np
import torch

print("=" * 70)
print("诊断韧性评估脚本")
print("=" * 70)
print()

# 加载模型
print("=== 加载 MamKO ===")
import args_new as new_args
args = dict(new_args.args, **new_args.ENV_PARAMS['waste_water'])
args['method'] = 'mamba'
args['env'] = 'waste_water'

from waste_water_system import waste_water_system as dreamer
env = dreamer()
env = env.unwrapped
args['state_dim'] = env.observation_space.shape[0]
args['act_dim'] = env.action_space.shape[0]
args['control'] = False

fold_path = 'save_model/mamba/waste_water'
args['save_model_path'] = fold_path + '/model.pt'
args['save_opti_path'] = fold_path + '/opti.pt'
args['shift_x'] = fold_path + '/shift_x.txt'
args['scale_x'] = fold_path + '/scale_x.txt'
args['shift_u'] = fold_path + '/shift_u.txt'
args['scale_u'] = fold_path + '/scale_u.txt'

from MamKO import Koopman_Desko as MamKO_Model
mamko = MamKO_Model(args)
mamko.parameter_restore(args)
print("✓ MamKO 加载成功")
print()

# 加载标准化参数
shift_x = np.loadtxt('save_model/mamba/waste_water/shift_x.txt')
scale_x = np.loadtxt('save_model/mamba/waste_water/scale_x.txt')
scale_x[np.where(scale_x == 0)] = 1

top_dims = np.argsort(scale_x)[-5:]
print(f"关键维度: {top_dims}")
print(f"对应标准差: {scale_x[top_dims]}")
print()

# 生成测试数据
print("=== 生成测试数据 ===")
from replay_fouling import ReplayMemory
replay_memory = ReplayMemory(args, env, predict_evolution=True)
test_dataset = replay_memory.dataset_test_draw

x_test, u_test = test_dataset[0]
if isinstance(x_test, np.ndarray):
    x_test = torch.from_numpy(x_test).float()
if isinstance(u_test, np.ndarray):
    u_test = torch.from_numpy(u_test).float()
x_test = x_test.unsqueeze(0)
u_test = u_test.unsqueeze(0)

print(f"测试数据形状: x={x_test.shape}, u={u_test.shape}")
print(f"x 范围: [{x_test.min():.4f}, {x_test.max():.4f}]")
print(f"u 范围: [{u_test.min():.4f}, {u_test.max():.4f}]")
print()

# 测试预测
print("=== 测试预测 ===")
old_horizon = args['old_horizon']
pred_horizon = args['pred_horizon']
print(f"old_horizon: {old_horizon}")
print(f"pred_horizon: {pred_horizon}")
print()

pred, _ = mamko.pred_forward_test(x_test, u_test, False, args, 0)
print(f"预测输出形状: {pred.shape}")
print(f"预测输出类型: {type(pred)}")

if pred.shape[0] != 1:
    pred = np.transpose(pred, (1, 0, 2))
    print(f"转置后形状: {pred.shape}")

print()

# 提取真实值和预测值
truth_norm = x_test.numpy()[0, old_horizon:old_horizon+pred_horizon, :]
pred_norm = pred[0, :pred_horizon, :]

print(f"truth_norm 形状: {truth_norm.shape}")
print(f"pred_norm 形状: {pred_norm.shape}")
print(f"truth_norm 范围: [{truth_norm.min():.4f}, {truth_norm.max():.4f}]")
print(f"pred_norm 范围: [{pred_norm.min():.4f}, {pred_norm.max():.4f}]")
print()

# 反标准化
print("=== 反标准化 ===")
truth_denorm = truth_norm * scale_x + shift_x
pred_denorm = pred_norm * scale_x + shift_x

print(f"truth_denorm 形状: {truth_denorm.shape}")
print(f"pred_denorm 形状: {pred_denorm.shape}")
print()

# 计算关键维度的误差
print("=== 关键维度误差 ===")
print(f"{'维度':<8} {'真实均值':<12} {'预测均值':<12} {'RMSE':<12} {'相对RMSE':<12}")
print("-" * 70)

errors = []
for dim in top_dims:
    truth_vals = truth_denorm[:, dim]
    pred_vals = pred_denorm[:, dim]

    truth_mean = np.mean(truth_vals)
    pred_mean = np.mean(pred_vals)
    rmse = np.sqrt(np.mean((pred_vals - truth_vals)**2))
    mean_val = np.mean(np.abs(truth_vals))
    rel_rmse = (rmse / mean_val * 100) if mean_val > 0 else 0

    print(f"{dim:<8} {truth_mean:<12.2f} {pred_mean:<12.2f} {rmse:<12.2f} {rel_rmse:<12.2f}%")
    errors.append(rel_rmse)

print()
print(f"平均相对RMSE: {np.mean(errors):.2f}%")
print()

# 诊断: 检查标准化是否合理
print("=== 诊断信息 ===")
print(f"1. 关键维度的 scale_x 值:")
for dim in top_dims:
    print(f"   维度 {dim}: scale={scale_x[dim]:.2f}, shift={shift_x[dim]:.2f}")
print()

print(f"2. 标准化数据的统计:")
print(f"   truth_norm 均值: {np.mean(truth_norm):.4f}")
print(f"   truth_norm 标准差: {np.std(truth_norm):.4f}")
print(f"   pred_norm 均值: {np.mean(pred_norm):.4f}")
print(f"   pred_norm 标准差: {np.std(pred_norm):.4f}")
print()

print(f"3. 反标准化数据的统计:")
print(f"   truth_denorm 均值: {np.mean(truth_denorm):.2f}")
print(f"   truth_denorm 标准差: {np.std(truth_denorm):.2f}")
print(f"   pred_denorm 均值: {np.mean(pred_denorm):.2f}")
print(f"   pred_denorm 标准差: {np.std(pred_denorm):.2f}")
print()

# 检查预测和真实值的差异
print(f"4. 预测误差分析:")
diff_norm = pred_norm - truth_norm
diff_denorm = pred_denorm - truth_denorm
print(f"   标准化空间误差: 均值={np.mean(diff_norm):.4f}, 标准差={np.std(diff_norm):.4f}")
print(f"   原始空间误差: 均值={np.mean(diff_denorm):.2f}, 标准差={np.std(diff_denorm):.2f}")
print()

# 与之前的结果对比
print("=" * 70)
print("【对比分析】")
print("=" * 70)
print(f"当前测试结果: {np.mean(errors):.2f}%")
print(f"之前 MamKO 结果: 14.6%")
print(f"之前 KoVAE v1 结果: 4391.7%")
print()

if np.mean(errors) > 100:
    print("⚠️  当前误差异常高！可能的原因:")
    print("  1. 测试数据与之前不同")
    print("  2. 模型加载有问题")
    print("  3. 标准化参数不匹配")
    print("  4. 预测方法调用不正确")
elif np.mean(errors) < 30:
    print("✓ 误差在合理范围内")
else:
    print("⚠️ 误差偏高但可能合理")
print()
