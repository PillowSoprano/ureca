import numpy as np
import torch
import matplotlib.pyplot as plt
from waste_water_system import waste_water_system
from MamKO import Koopman_Desko
import args_new as new_args

# 加载配置
args = dict(new_args.args, **new_args.ENV_PARAMS['waste_water'])
args['env'] = 'waste_water'
args['method'] = 'mamba'
args['control'] = False  # 只做预测，不做控制

# 创建环境
env = waste_water_system().unwrapped
args['state_dim'] = env.observation_space.shape[0]
args['act_dim'] = env.action_space.shape[0]

# 设置模型路径
fold_path = 'save_model/mamba/waste_water'
args['save_model_path'] = f'{fold_path}/model.pt'
args['save_opti_path'] = f'{fold_path}/opti.pt'
args['shift_x'] = f'{fold_path}/shift_x.txt'
args['scale_x'] = f'{fold_path}/scale_x.txt'
args['shift_u'] = f'{fold_path}/shift_u.txt'
args['scale_u'] = f'{fold_path}/scale_u.txt'

print("加载模型...")
model = Koopman_Desko(args)
model.parameter_restore(args)
model.net.eval()

print("生成测试轨迹...")
# 生成一条测试轨迹
state = env.reset()
states = [state]
actions = []

for t in range(200):  # 生成200步
    action = env.get_action()  # 使用环境的默认动作
    actions.append(action)
    state, reward, done, info = env.step(action)
    states.append(state)
    if done:
        break

states = np.array(states)
actions = np.array(actions)

print(f"轨迹形状: states={states.shape}, actions={actions.shape}")

# 加载归一化参数
shift_x = np.loadtxt(args['shift_x'])
scale_x = np.loadtxt(args['scale_x'])
shift_u = np.loadtxt(args['shift_u'])
scale_u = np.loadtxt(args['scale_u'])

# 归一化
states_norm = (states - shift_x) / scale_x
actions_norm = (actions - shift_u) / scale_u

# 找出关键维度
top_dims = np.argsort(scale_x)[-5:]
print(f"关键维度: {top_dims}")
print(f"对应标准差: {scale_x[top_dims]}")

# 准备模型输入（使用滑动窗口）
old_horizon = args['old_horizon']
pred_horizon = args['pred_horizon']

print(f"\n使用 old_horizon={old_horizon}, pred_horizon={pred_horizon}")
print("开始预测...")

# 单步预测
predictions = []
ground_truth = []

with torch.no_grad():
    for i in range(old_horizon, len(states_norm) - pred_horizon):
        # 输入：过去old_horizon步的状态和动作
        x_in = states_norm[i-old_horizon:i+pred_horizon]
        u_in = actions_norm[i-old_horizon:i+pred_horizon-1]
        
        # 转换为torch tensor
        x_tensor = torch.FloatTensor(x_in).unsqueeze(0)  # [1, time, dim]
        u_tensor = torch.FloatTensor(u_in).unsqueeze(0)
        
        # 预测
        loss, pred = model.net(x_tensor, u_tensor)
        
        # 记录预测和真实值（只记录未来pred_horizon步的第一步）
        if isinstance(pred, torch.Tensor):
            predictions.append(pred[0, 0, :].cpu().numpy())
        else:
            predictions.append(pred[0, 0, :])  # [dim]
        ground_truth.append(states_norm[i])

predictions = np.array(predictions)
ground_truth = np.array(ground_truth)

print(f"预测形状: {predictions.shape}")
print(f"真实值形状: {ground_truth.shape}")

# 反归一化
pred_denorm = predictions * scale_x + shift_x
truth_denorm = ground_truth * scale_x + shift_x

# 绘制关键维度
fig, axes = plt.subplots(5, 1, figsize=(15, 12))
fig.suptitle('MamKO在关键维度上的预测效果', fontsize=16)

for idx, dim in enumerate(top_dims):
    ax = axes[idx]
    
    time_steps = np.arange(len(truth_denorm))
    ax.plot(time_steps, truth_denorm[:, dim], 'k-', label='真实值', linewidth=2)
    ax.plot(time_steps, pred_denorm[:, dim], 'r--', label='预测值', linewidth=2, alpha=0.7)
    
    # 计算误差
    rmse = np.sqrt(np.mean((pred_denorm[:, dim] - truth_denorm[:, dim])**2))
    mae = np.mean(np.abs(pred_denorm[:, dim] - truth_denorm[:, dim]))
    
    # 计算相对误差
    mean_val = np.mean(np.abs(truth_denorm[:, dim]))
    rel_rmse = (rmse / mean_val * 100) if mean_val > 0 else 0
    
    ax.set_title(f'维度 {dim} | RMSE={rmse:.2f} ({rel_rmse:.1f}%), MAE={mae:.2f}')
    ax.set_xlabel('时间步')
    ax.set_ylabel('数值')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mamko_key_dims_test.png', dpi=150, bbox_inches='tight')
print(f"\n✅ 测试结果已保存到: mamko_key_dims_test.png")

# 打印性能指标
print("\n=== 关键维度性能总结 ===")
for dim in top_dims:
    rmse = np.sqrt(np.mean((pred_denorm[:, dim] - truth_denorm[:, dim])**2))
    mae = np.mean(np.abs(pred_denorm[:, dim] - truth_denorm[:, dim]))
    mean_val = np.mean(np.abs(truth_denorm[:, dim]))
    rel_rmse = (rmse / mean_val * 100) if mean_val > 0 else 0
    
    print(f"维度 {dim}: RMSE={rmse:.2f}, MAE={mae:.2f}, 相对RMSE={rel_rmse:.1f}%")
