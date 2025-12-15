import numpy as np
import matplotlib.pyplot as plt

print("加载数据...")
# 加载预测和真实数据
try:
    x_pred = np.load('Prediction/mamba/x_pred.npy')
    x_real = np.load('Prediction/mamba/x_.npy') 
    print(f"预测数据形状: {x_pred.shape}")
    print(f"真实数据形状: {x_real.shape}")
except:
    print("找不到预测数据文件，使用训练过程中的最后一个epoch的图")
    import sys
    sys.exit(0)

# 加载归一化参数
shift_x = np.loadtxt('save_model/mamba/waste_water/shift_x.txt')
scale_x = np.loadtxt('save_model/mamba/waste_water/scale_x.txt')

# 找出最重要的5个维度
top_dims = np.argsort(scale_x)[-5:]
print(f"\n最重要的5个维度: {top_dims}")
print(f"对应标准差: {scale_x[top_dims]}")

# 反归一化
x_pred_denorm = x_pred * scale_x + shift_x
x_real_denorm = x_real * scale_x + shift_x

# 为每个重要维度绘制预测 vs 真实
fig, axes = plt.subplots(5, 1, figsize=(15, 12))
fig.suptitle('关键维度预测效果 (标准差最大的5个维度)', fontsize=16)

for idx, dim in enumerate(top_dims):
    ax = axes[idx]
    
    # 取第一个轨迹的预测
    if len(x_pred_denorm.shape) == 4:  # (n_trajectories, batch, time, dim)
        pred_vals = x_pred_denorm[0, 0, :, dim]
        real_vals = x_real_denorm[0, :, dim]
    elif len(x_pred_denorm.shape) == 3:  # (batch, time, dim)
        pred_vals = x_pred_denorm[0, :, dim]
        real_vals = x_real_denorm[0, :, dim]
    
    # 绘图
    time_steps = np.arange(len(real_vals))
    ax.plot(time_steps, real_vals, 'k-', label='真实值', linewidth=2)
    
    # 如果预测长度小于真实值，只画对应部分
    pred_time = time_steps[:len(pred_vals)]
    ax.plot(pred_time, pred_vals, 'r--', label='预测值', linewidth=2, alpha=0.7)
    
    # 计算误差
    min_len = min(len(pred_vals), len(real_vals))
    rmse = np.sqrt(np.mean((pred_vals[:min_len] - real_vals[:min_len])**2))
    mae = np.mean(np.abs(pred_vals[:min_len] - real_vals[:min_len]))
    
    ax.set_title(f'维度 {dim} (std={scale_x[dim]:.1f}) | RMSE={rmse:.2f}, MAE={mae:.2f}')
    ax.set_xlabel('时间步')
    ax.set_ylabel('数值')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('key_dimensions_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n✅ 图表已保存到: key_dimensions_analysis.png")

# 计算总体指标
print("\n=== 关键维度性能指标 ===")
for dim in top_dims:
    if len(x_pred_denorm.shape) == 4:
        pred_vals = x_pred_denorm[0, 0, :, dim]
        real_vals = x_real_denorm[0, :, dim]
    else:
        pred_vals = x_pred_denorm[0, :, dim]
        real_vals = x_real_denorm[0, :, dim]
    
    min_len = min(len(pred_vals), len(real_vals))
    rmse = np.sqrt(np.mean((pred_vals[:min_len] - real_vals[:min_len])**2))
    mae = np.mean(np.abs(pred_vals[:min_len] - real_vals[:min_len]))
    mape = np.mean(np.abs((pred_vals[:min_len] - real_vals[:min_len]) / (real_vals[:min_len] + 1e-8))) * 100
    
    print(f"维度 {dim}: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")
