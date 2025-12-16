import numpy as np
import torch
import matplotlib.pyplot as plt
from waste_water_system import waste_water_system
from MamKO import Koopman_Desko as MamKO_Model
from kovae_model import Koopman_Desko as KoVAE_Model
import args_new as new_args

print("=" * 60)
print("MamKO vs KoVAE 性能对比")
print("=" * 60)

# 加载基本配置
args = dict(new_args.args, **new_args.ENV_PARAMS['waste_water'])
args['env'] = 'waste_water'
args['control'] = False

env = waste_water_system().unwrapped
args['state_dim'] = env.observation_space.shape[0]
args['act_dim'] = env.action_space.shape[0]

# 加载归一化参数（两个模型共享）
shift_x = np.loadtxt('save_model/mamba/waste_water/shift_x.txt')
scale_x = np.loadtxt('save_model/mamba/waste_water/scale_x.txt')

# 找出关键维度
top_dims = np.argsort(scale_x)[-5:]
print(f"\n关键维度 (标准差最大的5个): {top_dims}")
print(f"对应标准差: {scale_x[top_dims]}\n")

# ========== 加载MamKO模型 ==========
print("加载 MamKO 模型...")
args_mamko = dict(args)
args_mamko['method'] = 'mamba'
fold_path_mamko = 'save_model/mamba/waste_water'
args_mamko['save_model_path'] = f'{fold_path_mamko}/model.pt'
args_mamko['save_opti_path'] = f'{fold_path_mamko}/opti.pt'
args_mamko['shift_x'] = f'{fold_path_mamko}/shift_x.txt'
args_mamko['scale_x'] = f'{fold_path_mamko}/scale_x.txt'
args_mamko['shift_u'] = f'{fold_path_mamko}/shift_u.txt'
args_mamko['scale_u'] = f'{fold_path_mamko}/scale_u.txt'

mamko = MamKO_Model(args_mamko)
mamko.parameter_restore(args_mamko)
mamko.net.eval()

# ========== 加载KoVAE模型 ==========
print("加载 KoVAE 模型...")
args_kovae = dict(args)
args_kovae['method'] = 'kovae'
args_kovae['z_dim'] = 64
args_kovae['h_dim'] = 256
args_kovae['alpha'] = 0.1
args_kovae['beta'] = 1e-3
args_kovae['gamma'] = 0.0
args_kovae['use_action'] = False
fold_path_kovae = 'save_model/kovae/waste_water'
args_kovae['save_model_path'] = f'{fold_path_kovae}/model.pt'
args_kovae['save_opti_path'] = f'{fold_path_kovae}/opti.pt'
args_kovae['shift_x'] = f'{fold_path_kovae}/shift_x.txt'
args_kovae['scale_x'] = f'{fold_path_kovae}/scale_x.txt'
args_kovae['shift_u'] = f'{fold_path_kovae}/shift_u.txt'
args_kovae['scale_u'] = f'{fold_path_kovae}/scale_u.txt'

kovae = KoVAE_Model(args_kovae)
kovae.parameter_restore(args_kovae)
kovae.model.eval()

# ========== 生成测试轨迹 ==========
print("\n生成测试轨迹...")
env.reset()
states = [env.reset()]
actions = []

for t in range(200):
    action = env.get_action()
    actions.append(action)
    state, _, done, _ = env.step(action)
    states.append(state)
    if done:
        break

states = np.array(states)
actions = np.array(actions)
print(f"轨迹形状: states={states.shape}, actions={actions.shape}")

# 归一化
shift_u = np.loadtxt(args_mamko['shift_u'])
scale_u = np.loadtxt(args_mamko['scale_u'])
states_norm = (states - shift_x) / scale_x
actions_norm = (actions - shift_u) / scale_u

# ========== MamKO预测 ==========
print("\nMamKO预测中...")
old_horizon = args_mamko['old_horizon']
pred_horizon = args_mamko['pred_horizon']

mamko_preds = []
ground_truth = []

with torch.no_grad():
    for i in range(old_horizon, len(states_norm) - pred_horizon):
        x_in = states_norm[i-old_horizon:i+pred_horizon]
        u_in = actions_norm[i-old_horizon:i+pred_horizon-1]
        
        x_tensor = torch.FloatTensor(x_in).unsqueeze(0)
        u_tensor = torch.FloatTensor(u_in).unsqueeze(0)
        
        _, pred = mamko.net(x_tensor, u_tensor)
        
        if isinstance(pred, torch.Tensor):
            mamko_preds.append(pred[0, 0, :].cpu().numpy())
        else:
            mamko_preds.append(pred[0, 0, :])
        ground_truth.append(states_norm[i])

mamko_preds = np.array(mamko_preds)
ground_truth = np.array(ground_truth)

# ========== KoVAE预测 ==========
print("KoVAE预测中...")
kovae_preds = []

with torch.no_grad():
    for i in range(old_horizon, len(states_norm) - pred_horizon):
        x_in = states_norm[i-old_horizon:i+pred_horizon]
        u_in = actions_norm[i-old_horizon:i+pred_horizon-1]
        
        x_tensor = torch.FloatTensor(x_in).unsqueeze(0)
        u_tensor = torch.FloatTensor(u_in).unsqueeze(0)
        
        # KoVAE的forward接口不同
        x_batch, u_batch, xin, _, _ = kovae._make_batch((x_tensor, u_tensor))
        _, xhat, _ = kovae.model(xin, kovae.alpha, kovae.beta, kovae.gamma)
        
        # 取第一个预测步
        kovae_preds.append(xhat[0, old_horizon, :].cpu().numpy())

kovae_preds = np.array(kovae_preds)

# 反归一化
mamko_denorm = mamko_preds * scale_x + shift_x
kovae_denorm = kovae_preds * scale_x + shift_x
truth_denorm = ground_truth * scale_x + shift_x

# ========== 计算性能指标 ==========
print("\n" + "=" * 60)
print("关键维度性能对比")
print("=" * 60)
print(f"{'维度':<8} {'MamKO RMSE':<15} {'KoVAE RMSE':<15} {'相对改进':<10}")
print("-" * 60)

for dim in top_dims:
    mamko_rmse = np.sqrt(np.mean((mamko_denorm[:, dim] - truth_denorm[:, dim])**2))
    kovae_rmse = np.sqrt(np.mean((kovae_denorm[:, dim] - truth_denorm[:, dim])**2))
    
    mean_val = np.mean(np.abs(truth_denorm[:, dim]))
    mamko_rel = (mamko_rmse / mean_val * 100) if mean_val > 0 else 0
    kovae_rel = (kovae_rmse / mean_val * 100) if mean_val > 0 else 0
    
    improvement = ((mamko_rmse - kovae_rmse) / mamko_rmse * 100) if mamko_rmse > 0 else 0
    
    print(f"{dim:<8} {mamko_rmse:>7.1f}({mamko_rel:>4.1f}%) {kovae_rmse:>7.1f}({kovae_rel:>4.1f}%) {improvement:>+6.1f}%")

# 计算整体平均
all_mamko_rmse = np.sqrt(np.mean((mamko_denorm - truth_denorm)**2))
all_kovae_rmse = np.sqrt(np.mean((kovae_denorm - truth_denorm)**2))
print("-" * 60)
print(f"{'整体':<8} {all_mamko_rmse:>14.1f} {all_kovae_rmse:>14.1f} {((all_mamko_rmse - all_kovae_rmse) / all_mamko_rmse * 100):>+6.1f}%")

# ========== 绘制对比图 ==========
print("\n生成对比图...")
fig, axes = plt.subplots(5, 1, figsize=(15, 15))
fig.suptitle('MamKO vs KoVAE: 关键维度预测对比', fontsize=16)

for idx, dim in enumerate(top_dims):
    ax = axes[idx]
    
    time_steps = np.arange(len(truth_denorm))
    ax.plot(time_steps, truth_denorm[:, dim], 'k-', label='真实值', linewidth=2)
    ax.plot(time_steps, mamko_denorm[:, dim], 'r--', label='MamKO', linewidth=2, alpha=0.7)
    ax.plot(time_steps, kovae_denorm[:, dim], 'b:', label='KoVAE', linewidth=2, alpha=0.7)
    
    mamko_rmse = np.sqrt(np.mean((mamko_denorm[:, dim] - truth_denorm[:, dim])**2))
    kovae_rmse = np.sqrt(np.mean((kovae_denorm[:, dim] - truth_denorm[:, dim])**2))
    mean_val = np.mean(np.abs(truth_denorm[:, dim]))
    mamko_rel = (mamko_rmse / mean_val * 100) if mean_val > 0 else 0
    kovae_rel = (kovae_rmse / mean_val * 100) if mean_val > 0 else 0
    
    ax.set_title(f'Dim {dim} | MamKO: {mamko_rel:.1f}% | KoVAE: {kovae_rel:.1f}%')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mamko_vs_kovae_comparison.png', dpi=150, bbox_inches='tight')
print("✅ 对比图已保存到: mamko_vs_kovae_comparison.png")

# ========== 加载损失曲线对比 ==========
print("\n生成损失曲线对比...")
mamko_loss_train = np.loadtxt('loss/mamba/waste_water/0/loss_.txt')
mamko_loss_test = np.loadtxt('loss/mamba/waste_water/0/loss_t.txt')

try:
    kovae_loss_train = np.loadtxt('loss/kovae/waste_water/0/loss_.txt')
    kovae_loss_test = np.loadtxt('loss/kovae/waste_water/0/loss_t.txt')
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 训练损失
    axes[0].plot(mamko_loss_train, 'r-', label='MamKO Train', linewidth=2)
    axes[0].plot(kovae_loss_train, 'b-', label='KoVAE Train', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # 测试损失
    axes[1].plot(mamko_loss_test, 'r-', label='MamKO Test', linewidth=2)
    axes[1].plot(kovae_loss_test, 'b-', label='KoVAE Test', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Test Loss Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('loss_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ 损失曲线对比已保存到: loss_comparison.png")
    
    print(f"\n最终训练损失: MamKO={mamko_loss_train[-1]:.6f}, KoVAE={kovae_loss_train[-1]:.6f}")
    print(f"最终测试损失: MamKO={mamko_loss_test[-1]:.6f}, KoVAE={kovae_loss_test[-1]:.6f}")
except:
    print("⚠️  KoVAE损失文件未找到，跳过损失曲线对比")

print("\n" + "=" * 60)
print("分析完成！")
print("=" * 60)
