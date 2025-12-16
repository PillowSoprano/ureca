#!/usr/bin/env python3
"""
快速测试基线误差 - 使用保存的测试数据
"""

import numpy as np
import torch

print("加载 MamKO...")
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

# 加载标准化参数
shift_x = np.loadtxt('save_model/mamba/waste_water/shift_x.txt')
scale_x = np.loadtxt('save_model/mamba/waste_water/scale_x.txt')
scale_x[np.where(scale_x == 0)] = 1
top_dims = np.argsort(scale_x)[-5:]

# 加载保存的测试数据
print("加载保存的测试数据...")
test_draw = torch.load(new_args.args['SAVE_DRAW'])
x_test, u_test = test_draw[0]
x_test = x_test.unsqueeze(0)
u_test = u_test.unsqueeze(0)

print(f"测试数据形状: x={x_test.shape}, u={u_test.shape}")

# 预测
old_horizon = args['old_horizon']
pred_horizon = args['pred_horizon']

pred, _ = mamko.pred_forward_test(x_test, u_test, False, args, 0)
if pred.shape[0] != 1:
    pred = np.transpose(pred, (1, 0, 2))

truth_norm = x_test.numpy()[0, old_horizon:old_horizon+pred_horizon, :]
pred_norm = pred[0, :pred_horizon, :]

# 反标准化
truth_denorm = truth_norm * scale_x + shift_x
pred_denorm = pred_norm * scale_x + shift_x

# 计算误差
errors = []
for dim in top_dims:
    rmse = np.sqrt(np.mean((pred_denorm[:, dim] - truth_denorm[:, dim])**2))
    mean_val = np.mean(np.abs(truth_denorm[:, dim]))
    rel_rmse = (rmse / mean_val * 100) if mean_val > 0 else 0
    errors.append(rel_rmse)
    print(f"维度 {dim}: {rel_rmse:.2f}%")

print(f"\n平均相对误差: {np.mean(errors):.2f}%")
print(f"(应该接近 14.6%)")
