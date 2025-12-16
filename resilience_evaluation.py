#!/usr/bin/env python3
"""
MamKO 韧性评估脚本
Resilience Evaluation for MamKO on Wastewater Treatment System

评估内容:
1. 噪声鲁棒性 (Noise Robustness)
2. 长期预测稳定性 (Long-term Stability)
3. 传感器故障容错 (Sensor Fault Tolerance)
4. 初始条件敏感性 (Initial Condition Sensitivity)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

print("=" * 70)
print("MamKO 韧性评估")
print("=" * 70)
print()

# =========================
# 加载 MamKO 模型
# =========================
print("=== 1. 加载模型 ===")
import args_new as new_args
args = dict(new_args.args, **new_args.ENV_PARAMS['waste_water'])
args['method'] = 'mamba'
args['env'] = 'waste_water'
args['continue_training'] = False

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
print("✓ MamKO 模型加载成功")
print()

# =========================
# 加载数据和标准化参数
# =========================
print("=== 2. 加载数据 ===")
shift_x = np.loadtxt('save_model/mamba/waste_water/shift_x.txt')
scale_x = np.loadtxt('save_model/mamba/waste_water/scale_x.txt')
shift_u = np.loadtxt('save_model/mamba/waste_water/shift_u.txt')
scale_u = np.loadtxt('save_model/mamba/waste_water/scale_u.txt')

# 防止除以零
scale_x[np.where(scale_x == 0)] = 1
scale_u[np.where(scale_u == 0)] = 1

# 关键维度
top_dims = np.argsort(scale_x)[-5:]
print(f"关键维度: {top_dims}")
print(f"对应标准差: {scale_x[top_dims]}")
print()

# 加载测试数据（使用与对比实验相同的数据）
print("加载测试数据...")
draw_path = new_args.args.get('SAVE_DRAW', 'save_model/mamba/waste_water/draw.pt')

# 如果配置的路径无效（如 /draw.pt），使用本地路径
if not draw_path or draw_path.startswith('/') and not os.access(os.path.dirname(draw_path) or '/', os.W_OK):
    draw_path = 'save_model/mamba/waste_water/draw.pt'
    print(f"  使用本地路径: {draw_path}")

# 如果测试数据文件不存在，生成一个并保存
if not os.path.exists(draw_path):
    print("  测试数据文件不存在，从 replay memory 生成...")
    from replay_fouling import ReplayMemory
    replay_memory = ReplayMemory(args, env, predict_evolution=True)
    test_draw = replay_memory.dataset_test_draw
    # 保存测试数据以便后续使用
    os.makedirs(os.path.dirname(draw_path), exist_ok=True)
    torch.save(test_draw, draw_path)
    print(f"  ✓ 生成并保存了 {len(test_draw)} 个测试样本到 {draw_path}")
else:
    test_draw = torch.load(draw_path)
    print(f"  ✓ 从文件加载了 {len(test_draw)} 个测试样本")
print()

# 使用多个测试样本
num_test_samples = min(5, len(test_draw))
test_samples = []
for i in range(num_test_samples):
    x, u = test_draw[i]
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if isinstance(u, np.ndarray):
        u = torch.from_numpy(u).float()
    test_samples.append((x.unsqueeze(0), u.unsqueeze(0)))

print(f"使用 {num_test_samples} 个测试样本进行评估")
print()

# =========================
# 辅助函数
# =========================
def denormalize(x_norm, shift, scale):
    """反标准化"""
    return x_norm * scale + shift

def calculate_error(pred, truth, shift, scale):
    """计算预测误差"""
    pred_denorm = denormalize(pred, shift, scale)
    truth_denorm = denormalize(truth, shift, scale)

    errors = {}
    for dim in top_dims:
        rmse = np.sqrt(np.mean((pred_denorm[:, dim] - truth_denorm[:, dim])**2))
        mean_val = np.mean(np.abs(truth_denorm[:, dim]))
        rel_rmse = (rmse / mean_val * 100) if mean_val > 0 else 0
        errors[dim] = {'rmse': rmse, 'rel_rmse': rel_rmse}

    avg_rel_rmse = np.mean([e['rel_rmse'] for e in errors.values()])
    return errors, avg_rel_rmse

# =========================
# 1. 基线性能 (无扰动)
# =========================
print("=" * 70)
print("测试 1: 基线性能 (无扰动)")
print("=" * 70)

baseline_errors = []
for i, (x_test, u_test) in enumerate(test_samples):
    pred, _ = mamko.pred_forward_test(x_test, u_test, False, args, 0)

    # 转换格式
    if pred.shape[0] != 1:
        pred = np.transpose(pred, (1, 0, 2))

    old_horizon = args['old_horizon']
    pred_horizon = args['pred_horizon']

    truth_norm = x_test.numpy()[0, old_horizon:old_horizon+pred_horizon, :]
    pred_norm = pred[0, :pred_horizon, :]

    _, avg_error = calculate_error(pred_norm, truth_norm, shift_x, scale_x)
    baseline_errors.append(avg_error)

baseline_mean = np.mean(baseline_errors)
baseline_std = np.std(baseline_errors)
print(f"基线平均误差: {baseline_mean:.2f}% ± {baseline_std:.2f}%")
print()

# =========================
# 2. 噪声鲁棒性测试
# =========================
print("=" * 70)
print("测试 2: 噪声鲁棒性")
print("=" * 70)

noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]  # 噪声水平 (相对于标准差)
noise_results = {level: [] for level in noise_levels}

for noise_level in noise_levels:
    print(f"  噪声水平: {noise_level*100:.0f}% 标准差...")

    for x_test, u_test in test_samples:
        # 添加高斯噪声
        noise = torch.randn_like(x_test) * noise_level
        x_noisy = x_test + noise

        pred, _ = mamko.pred_forward_test(x_noisy, u_test, False, args, 0)

        if pred.shape[0] != 1:
            pred = np.transpose(pred, (1, 0, 2))

        truth_norm = x_test.numpy()[0, old_horizon:old_horizon+pred_horizon, :]
        pred_norm = pred[0, :pred_horizon, :]

        _, avg_error = calculate_error(pred_norm, truth_norm, shift_x, scale_x)
        noise_results[noise_level].append(avg_error)

print("\n噪声鲁棒性结果:")
print(f"{'噪声水平':<12} {'平均误差':<12} {'vs 基线':<12}")
print("-" * 40)
for level in noise_levels:
    mean_error = np.mean(noise_results[level])
    ratio = mean_error / baseline_mean
    print(f"{level*100:6.0f}%      {mean_error:8.2f}%    {ratio:6.2f}x")
print()

# =========================
# 3. 长期预测稳定性
# =========================
print("=" * 70)
print("测试 3: 长期预测稳定性")
print("=" * 70)

# 使用第一个测试样本做长期预测
x_test, u_test = test_samples[0]
long_term_horizons = [10, 20, 30, 50]
long_term_results = {}

for horizon in long_term_horizons:
    if old_horizon + horizon > x_test.shape[1]:
        print(f"  时域 {horizon}: 超出数据长度，跳过")
        continue

    print(f"  预测时域: {horizon} 步...")

    # 修改预测时域
    args_temp = args.copy()
    args_temp['pred_horizon'] = horizon

    pred, _ = mamko.pred_forward_test(x_test, u_test, False, args_temp, 0)

    if pred.shape[0] != 1:
        pred = np.transpose(pred, (1, 0, 2))

    truth_norm = x_test.numpy()[0, old_horizon:old_horizon+horizon, :]
    pred_norm = pred[0, :horizon, :]

    # 确保形状匹配（取最小长度）
    min_len = min(len(pred_norm), len(truth_norm))
    truth_norm = truth_norm[:min_len]
    pred_norm = pred_norm[:min_len]

    _, avg_error = calculate_error(pred_norm, truth_norm, shift_x, scale_x)
    long_term_results[horizon] = avg_error

print("\n长期预测结果:")
print(f"{'预测步数':<12} {'平均误差':<12} {'vs 基线':<12}")
print("-" * 40)
for horizon in sorted(long_term_results.keys()):
    error = long_term_results[horizon]
    ratio = error / baseline_mean
    print(f"{horizon:6d}       {error:8.2f}%    {ratio:6.2f}x")
print()

# =========================
# 4. 传感器故障容错
# =========================
print("=" * 70)
print("测试 4: 传感器故障容错")
print("=" * 70)

dropout_rates = [0.05, 0.1, 0.2, 0.3]  # 丢失比例
dropout_results = {rate: [] for rate in dropout_rates}

for dropout_rate in dropout_rates:
    print(f"  传感器故障率: {dropout_rate*100:.0f}%...")

    for x_test, u_test in test_samples:
        # 随机丢失一些维度的数据（设为0）
        x_dropout = x_test.clone()
        mask = torch.rand(x_test.shape[-1]) > dropout_rate
        x_dropout[:, :, ~mask] = 0  # 故障的传感器输出0

        pred, _ = mamko.pred_forward_test(x_dropout, u_test, False, args, 0)

        if pred.shape[0] != 1:
            pred = np.transpose(pred, (1, 0, 2))

        truth_norm = x_test.numpy()[0, old_horizon:old_horizon+pred_horizon, :]
        pred_norm = pred[0, :pred_horizon, :]

        _, avg_error = calculate_error(pred_norm, truth_norm, shift_x, scale_x)
        dropout_results[dropout_rate].append(avg_error)

print("\n传感器故障容错结果:")
print(f"{'故障率':<12} {'平均误差':<12} {'vs 基线':<12}")
print("-" * 40)
for rate in dropout_rates:
    mean_error = np.mean(dropout_results[rate])
    ratio = mean_error / baseline_mean
    print(f"{rate*100:6.0f}%      {mean_error:8.2f}%    {ratio:6.2f}x")
print()

# =========================
# 5. 可视化结果
# =========================
print("=" * 70)
print("生成可视化...")
print("=" * 70)

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

# 1. 噪声鲁棒性
ax1 = fig.add_subplot(gs[0, 0])
noise_means = [np.mean(noise_results[l]) for l in noise_levels]
noise_stds = [np.std(noise_results[l]) for l in noise_levels]
ax1.errorbar([l*100 for l in noise_levels], noise_means, yerr=noise_stds,
             marker='o', capsize=5, linewidth=2, markersize=8)
ax1.axhline(y=baseline_mean, color='r', linestyle='--', label='Baseline', linewidth=2)
ax1.set_xlabel('Noise Level (% of std)', fontsize=12)
ax1.set_ylabel('Relative RMSE (%)', fontsize=12)
ax1.set_title('Noise Robustness', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. 长期预测
ax2 = fig.add_subplot(gs[0, 1])
horizons = sorted(long_term_results.keys())
errors = [long_term_results[h] for h in horizons]
ax2.plot(horizons, errors, marker='s', linewidth=2, markersize=8)
ax2.axhline(y=baseline_mean, color='r', linestyle='--', label='Baseline (20 steps)', linewidth=2)
ax2.set_xlabel('Prediction Horizon (steps)', fontsize=12)
ax2.set_ylabel('Relative RMSE (%)', fontsize=12)
ax2.set_title('Long-term Prediction Stability', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# 3. 传感器故障容错
ax3 = fig.add_subplot(gs[1, 0])
dropout_means = [np.mean(dropout_results[r]) for r in dropout_rates]
dropout_stds = [np.std(dropout_results[r]) for r in dropout_rates]
ax3.errorbar([r*100 for r in dropout_rates], dropout_means, yerr=dropout_stds,
             marker='^', capsize=5, linewidth=2, markersize=8, color='orange')
ax3.axhline(y=baseline_mean, color='r', linestyle='--', label='Baseline', linewidth=2)
ax3.set_xlabel('Sensor Dropout Rate (%)', fontsize=12)
ax3.set_ylabel('Relative RMSE (%)', fontsize=12)
ax3.set_title('Sensor Fault Tolerance', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()

# 4. 综合对比
ax4 = fig.add_subplot(gs[1, 1])
categories = ['Baseline', 'Noise\n(10%)', 'Dropout\n(10%)', 'Long-term\n(50 steps)']
values = [
    baseline_mean,
    np.mean(noise_results[0.1]),
    np.mean(dropout_results[0.1]),
    long_term_results.get(50, baseline_mean * 2)
]
colors = ['green', 'blue', 'orange', 'purple']
bars = ax4.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax4.axhline(y=baseline_mean, color='r', linestyle='--', linewidth=2, label='Baseline')
ax4.set_ylabel('Relative RMSE (%)', fontsize=12)
ax4.set_title('Resilience Comparison', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
ax4.legend()

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 5. 韧性评分雷达图
ax5 = fig.add_subplot(gs[2, :], projection='polar')

# 计算各项韧性指标 (归一化到 0-1, 1 是最好)
def resilience_score(error, baseline):
    # 误差越接近基线，分数越高
    ratio = error / baseline
    return max(0, min(1, 2 - ratio))  # 2x基线 → 0分, 1x基线 → 1分

scores = {
    'Noise (10%)': resilience_score(np.mean(noise_results[0.1]), baseline_mean),
    'Noise (50%)': resilience_score(np.mean(noise_results[0.5]), baseline_mean),
    'Dropout (10%)': resilience_score(np.mean(dropout_results[0.1]), baseline_mean),
    'Dropout (30%)': resilience_score(np.mean(dropout_results[0.3]), baseline_mean),
    'Long-term (30)': resilience_score(long_term_results.get(30, baseline_mean*1.5), baseline_mean),
    'Long-term (50)': resilience_score(long_term_results.get(50, baseline_mean*2), baseline_mean),
}

categories_radar = list(scores.keys())
values_radar = list(scores.values())
values_radar += values_radar[:1]  # 闭合

angles = np.linspace(0, 2 * np.pi, len(categories_radar), endpoint=False).tolist()
angles += angles[:1]

ax5.plot(angles, values_radar, 'o-', linewidth=2, markersize=8, label='MamKO')
ax5.fill(angles, values_radar, alpha=0.25)
ax5.set_xticks(angles[:-1])
ax5.set_xticklabels(categories_radar, fontsize=10)
ax5.set_ylim(0, 1)
ax5.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax5.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
ax5.set_title('Resilience Score Radar', fontsize=14, fontweight='bold', pad=20)
ax5.grid(True)
ax5.legend(loc='upper right')

plt.suptitle('MamKO Resilience Evaluation Report', fontsize=16, fontweight='bold', y=0.98)

# 保存图片
os.makedirs('results', exist_ok=True)
plt.savefig('results/mamko_resilience_evaluation.png', dpi=300, bbox_inches='tight')
print("✓ 可视化已保存: results/mamko_resilience_evaluation.png")
print()

# =========================
# 6. 生成总结报告
# =========================
print("=" * 70)
print("总结报告")
print("=" * 70)
print()

print("【韧性评估结果】")
print()
print(f"1. 基线性能: {baseline_mean:.2f}% ± {baseline_std:.2f}%")
print()

print("2. 噪声鲁棒性:")
for level in noise_levels:
    mean_error = np.mean(noise_results[level])
    ratio = mean_error / baseline_mean
    status = "✓" if ratio < 1.5 else "⚠️" if ratio < 2.0 else "❌"
    print(f"   {level*100:4.0f}% 噪声: {mean_error:6.2f}% ({ratio:.2f}x 基线) {status}")
print()

print("3. 长期预测稳定性:")
for horizon in sorted(long_term_results.keys()):
    error = long_term_results[horizon]
    ratio = error / baseline_mean
    status = "✓" if ratio < 1.5 else "⚠️" if ratio < 2.5 else "❌"
    print(f"   {horizon:3d} 步: {error:6.2f}% ({ratio:.2f}x 基线) {status}")
print()

print("4. 传感器故障容错:")
for rate in dropout_rates:
    mean_error = np.mean(dropout_results[rate])
    ratio = mean_error / baseline_mean
    status = "✓" if ratio < 1.5 else "⚠️" if ratio < 2.0 else "❌"
    print(f"   {rate*100:4.0f}% 故障: {mean_error:6.2f}% ({ratio:.2f}x 基线) {status}")
print()

print("【评估标准】")
print("  ✓ 优秀: < 1.5x 基线")
print("  ⚠️ 良好: 1.5-2.0x 基线")
print("  ❌ 需改进: > 2.0x 基线")
print()

print("【结论】")
print("  MamKO 在废水处理系统建模中展现了:")
avg_noise_degradation = np.mean([np.mean(noise_results[l])/baseline_mean for l in noise_levels])
avg_dropout_degradation = np.mean([np.mean(dropout_results[r])/baseline_mean for r in dropout_rates])
print(f"  - 噪声鲁棒性: 平均性能下降 {(avg_noise_degradation-1)*100:.1f}%")
print(f"  - 故障容错性: 平均性能下降 {(avg_dropout_degradation-1)*100:.1f}%")
if 50 in long_term_results:
    print(f"  - 长期稳定性: 50步预测误差为基线的 {long_term_results[50]/baseline_mean:.2f}x")
print()
print("=" * 70)
print("韧性评估完成!")
print("=" * 70)
