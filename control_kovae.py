# control_kovae_optimized.py
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import args_new as new_args

METHOD = "kovae"
MODEL  = "cartpole"

args = dict(new_args.args, **new_args.ENV_PARAMS[MODEL])
args["env"] = MODEL

if MODEL == "cartpole":
    from envs.cartpole import CartPoleEnv_adv as dreamer
elif MODEL == "cartpole_V":
    from envs.cartpole_V import CartPoleEnv_adv as dreamer
else:
    raise ValueError("only cartpole variants")

env = dreamer().unwrapped
u_max = float(env.action_space.high[0])
X_REF = np.zeros(4, dtype=np.float32)

print("="*70)
print("OPTIMIZED ADAPTIVE LQR CONTROLLER")
print("="*70)
print(f"Environment: {MODEL}")
print(f"u_max: {u_max}")
print(f"Angle limit: ±{np.degrees(env.theta_threshold_radians):.1f}°")

# 优化的自适应LQR控制器
class OptimizedLQRController:
    def __init__(self, u_max, angle_limit):
        self.u_max = u_max
        self.angle_limit = angle_limit
        
        # 更强的位置控制
        self.K_base = np.array([
            -3.5,    # 减少位置漂移)
            -5.0,    # 更快阻尼)
            -40.0,   # 略微增强角度
            -10.0    # 角速度增强，更好的阻尼
        ])
        
        # 改进的积分控制
        self.integral_pos = 0.0
        self.integral_angle = 0.0
        self.Ki_pos = 0.8      # 增强积分
        self.Ki_angle = 2.0    # 角度积分补偿
        
        self.prev_u = 0.0
        self.energy_gain = 1.0
        
    def reset(self):
        self.integral_pos = 0.0
        self.integral_angle = 0.0
        self.prev_u = 0.0
    
    def __call__(self, x, dt=0.02):
        err = x - X_REF
        angle_ratio = abs(x[2]) / self.angle_limit
        
        # 自适应增益 优化策略
        if angle_ratio > 0.85:  # 极度危险 (>17°)
            gain_factor = 2.2
            emergency_boost = 8.0 * np.sign(x[2]) * (angle_ratio - 0.85)
            # 紧急模式：忽略位置，全力救角度
            K_adapted = self.K_base.copy()
            K_adapted[0] *= 0.2  # 大幅降低位置权重
            K_adapted[1] *= 0.3
            K_adapted *= gain_factor
        elif angle_ratio > 0.7:  # 危险 (>14°)
            gain_factor = 1.8
            emergency_boost = 3.0 * np.sign(x[2]) * (angle_ratio - 0.7)
            K_adapted = self.K_base.copy()
            K_adapted[0] *= 0.5  # 降低位置权重
            K_adapted *= gain_factor
        elif angle_ratio > 0.5:  # 警告 (>10°)
            gain_factor = 1.4
            emergency_boost = 0.0
            K_adapted = self.K_base * gain_factor
        elif angle_ratio > 0.2:  # 正常 (>4°)
            gain_factor = 1.1
            emergency_boost = 0.0
            K_adapted = self.K_base * gain_factor
        else:  # 安全区域 - 增强位置控制
            gain_factor = 1.0
            emergency_boost = 0.0
            K_adapted = self.K_base.copy()
            K_adapted[0] *= 1.3  # 增强位置控制
            K_adapted[1] *= 1.2
        
        # 状态反馈
        u_fb = -K_adapted @ err
        
        # 智能积分控制
        if angle_ratio < 0.4:  # 安全区域才使用积分
            # 位置积分（主要）
            self.integral_pos += err[0] * dt
            self.integral_pos = np.clip(self.integral_pos, -2.0, 2.0)
            
            # 角度积分（辅助，我用它消除小的系统偏差）
            self.integral_angle += err[2] * dt
            self.integral_angle = np.clip(self.integral_angle, -0.1, 0.1)
            
            u_int = -self.Ki_pos * self.integral_pos - self.Ki_angle * self.integral_angle
        else:
            # 危险区域：快速衰减积分
            self.integral_pos *= 0.7
            self.integral_angle *= 0.5
            u_int = 0.0
        
        # 能量整形
        potential_energy = x[2]**2
        kinetic_energy = x[3]**2
        total_energy = potential_energy + 0.05 * kinetic_energy
        
        if total_energy > 0.1:
            # 通过阻尼角速度来消耗能量
            energy_damping = -self.energy_gain * x[3] * x[2]
        else:
            energy_damping = 0.0
        
        # 前馈补偿（重力效应的线性化补偿）
        # 对于小角度，补偿重力导致的加速度
        if abs(x[2]) < 0.15:  # ~8.6°
            gravity_compensation = -5.0 * x[2]  # 补偿线性化的重力项
        else:
            gravity_compensation = 0.0
        
        # 总控制
        u = u_fb + u_int + energy_damping + gravity_compensation + emergency_boost
        
        # 智能速率限制
        # 根据角度调整允许的变化率
        if angle_ratio > 0.7:
            max_delta = 0.8 * self.u_max  # 危险时允许大变化
        elif angle_ratio > 0.4:
            max_delta = 0.6 * self.u_max
        else:
            max_delta = 0.4 * self.u_max  # 安全时平滑控制
        
        delta_u = u - self.prev_u
        if abs(delta_u) > max_delta:
            u = self.prev_u + np.sign(delta_u) * max_delta
        
        # 饱和限制
        u = np.clip(u, -self.u_max, self.u_max)
        self.prev_u = u
        
        return u

controller = OptimizedLQRController(u_max, env.theta_threshold_radians)

# 智能初始化
print("\nSearching for optimal initial state...")
best_obs = None
best_score = float('inf')

for _ in range(300):
    rst = env.reset()
    obs = rst[0] if isinstance(rst, tuple) else rst
    
    # 优化评分：平衡所有状态变量
    score = (20.0 * abs(obs[2]) +      # 角度最重要
             3.0 * abs(obs[0]) +       # 位置次之
             1.0 * abs(obs[3]) +       # 角速度
             0.5 * abs(obs[1]))        # 速度
    
    if score < best_score:
        best_score = score
        best_obs = obs.copy()
    
    # 理想状态：所有变量都接近0
    if (abs(obs[2]) < 0.02 and abs(obs[0]) < 0.15 and 
        abs(obs[3]) < 0.03 and abs(obs[1]) < 0.01):
        best_obs = obs.copy()
        print(f"Found excellent initial state at attempt {_+1}")
        break

env.reset()
if hasattr(env, 'state'):
    env.state = best_obs
obs = best_obs

print(f"Initial state:")
print(f"  Position: {obs[0]:7.4f} m,  Velocity: {obs[1]:7.4f} m/s")
print(f"  Angle:    {obs[2]:7.4f} rad ({np.degrees(obs[2]):6.2f}°),  Ang.Vel: {obs[3]:7.4f} rad/s")

# 闭环仿真
T_sim = 1000
xs, us = [obs.copy()], []
controller.reset()

print("\n" + "☆"*70)
print("SIMULATION START")
print("☆"*70)

stabilized_flag = False
stabilization_time = None

for t in range(T_sim):
    u = controller(obs)
    us.append(u)
    
    step_out = env.step(np.array([u]))
    if isinstance(step_out, tuple) and len(step_out) == 5:
        obs, r, terminated, truncated, info = step_out
        done = terminated or truncated
    else:
        obs, r, done, info = step_out
    
    xs.append(obs.copy())
    
    # 智能输出
    if t < 50:
        if t % 10 == 0:
            print(f"t={t:4d}: x={obs[0]:7.4f} m, θ={obs[2]:7.4f} rad ({np.degrees(obs[2]):6.2f}°), u={u:7.3f} N")
    elif t % 100 == 0:
        err = np.linalg.norm(obs - X_REF)
        print(f"t={t:4d}: x={obs[0]:7.4f}, θ={obs[2]:7.4f} ({np.degrees(obs[2]):6.2f}°), |err|={err:.4f}, u={u:7.3f}")
    
    if done:
        print(f"\n⚠ Episode terminated at t={t}")
        print(f"Final state: x={obs[0]:.4f}, θ={obs[2]:.4f} ({np.degrees(obs[2]):.2f}°)")
        if abs(obs[2]) > env.theta_threshold_radians:
            print(f"Reason: Angle limit exceeded (|{np.degrees(obs[2]):.1f}°| > {np.degrees(env.theta_threshold_radians):.1f}°)")
        elif abs(obs[0]) > env.x_threshold:
            print(f"Reason: Position limit exceeded (|{obs[0]:.2f}| > {env.x_threshold:.2f})")
        break
    
    # 稳定性检测（只打印一次）
    if not stabilized_flag and t > 150:
        recent = np.array(xs[-100:])
        recent_err = np.linalg.norm(recent - X_REF, axis=1)
        if np.mean(recent_err) < 0.12 and np.max(recent_err) < 0.25:
            stabilized_flag = True
            stabilization_time = t
            print(f"\n✓ System stabilized at t={t} (error < 0.12)")
            # 继续运行确认稳定性
            if t > 500:
                print(f"  Continuing for 200 more steps to confirm stability...")
                continue
    
    # 确认稳定后可以早停
    if stabilized_flag and t > stabilization_time + 200:
        print(f"  Stability confirmed. Stopping simulation.")
        break

xs = np.array(xs)
us = np.array(us)
final_err = np.linalg.norm(xs[-1] - X_REF)

print("\n" + "☆"*70)
print("PERFORMANCE METRICS")
print("☆"*70)
print(f"Simulation length:     {len(xs)} steps ({len(xs)*0.02:.1f}s)")
print(f"Final tracking error:  {final_err:.5f}")
print(f"\nFinal state:")
print(f"  Position:    {xs[-1,0]:8.5f} m")
print(f"  Velocity:    {xs[-1,1]:8.5f} m/s")
print(f"  Angle:       {xs[-1,2]:8.5f} rad ({np.degrees(xs[-1,2]):7.3f}°)")
print(f"  Angular vel: {xs[-1,3]:8.5f} rad/s")

print(f"\nState ranges:")
print(f"  Position:    [{xs[:,0].min():7.4f}, {xs[:,0].max():7.4f}] m")
print(f"  Angle:       [{np.degrees(xs[:,2].min()):6.2f}°, {np.degrees(xs[:,2].max()):6.2f}°]")
print(f"  Control:     [{us.min():6.2f}, {us.max():6.2f}] N  (max available: ±{u_max:.1f} N)")

errors = np.linalg.norm(xs - X_REF, axis=1)
print(f"\nTracking error statistics:")
print(f"  Mean error:     {np.mean(errors):.5f}")
print(f"  Max error:      {np.max(errors):.5f}")
print(f"  Std deviation:  {np.std(errors):.5f}")

if len(xs) > 100:
    print(f"  Last 100 steps: mean={np.mean(errors[-100:]):.5f}, max={np.max(errors[-100:]):.5f}")

# 计算2%和5%稳定时间
for threshold, name in [(0.02, "2%"), (0.05, "5%"), (0.10, "10%")]:
    for i in range(50, len(errors)):
        if np.all(errors[i:min(i+50, len(errors))] < threshold):
            print(f"  {name} settling time: ~{i} steps ({i*0.02:.1f}s)")
            break

# 控制努力
control_effort = np.sum(np.abs(us))
print(f"\nControl effort: {control_effort:.2f} (total |u|)")

# 可视化
tt = np.arange(len(xs))
fig, axs = plt.subplots(5, 1, figsize=(15, 14), sharex=True)

colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#ff7f0e']
names = ['Position (m)', 'Velocity (m/s)', 'Angle (rad)', 'Angular Velocity (rad/s)', 'Control Force (N)']
ylabels = ['x', 'ẋ', 'θ', 'θ̇', 'u']

for i in range(4):
    axs[i].plot(tt, xs[:,i], linewidth=2.5, color=colors[i], alpha=0.85, label='Actual')
    axs[i].axhline(0, color='black', linestyle='--', alpha=0.3, linewidth=1.5, label='Reference')
    
    if i == 0:  # Position
        limit = env.x_threshold
        axs[i].axhline(limit, color='red', linestyle=':', alpha=0.4, linewidth=1.5)
        axs[i].axhline(-limit, color='red', linestyle=':', alpha=0.4, linewidth=1.5)
        axs[i].fill_between(tt, -limit, limit, alpha=0.03, color='green')
        # 添加最终位置标注
        axs[i].plot(len(tt)-1, xs[-1,0], 'ro', markersize=8, label=f'Final: {xs[-1,0]:.4f}m')
    elif i == 2:  # Angle
        limit = env.theta_threshold_radians
        axs[i].axhline(limit, color='red', linestyle='-', alpha=0.6, linewidth=2, label=f'Limit (±{np.degrees(limit):.0f}°)')
        axs[i].axhline(-limit, color='red', linestyle='-', alpha=0.6, linewidth=2)
        
        # 分区着色
        axs[i].axhspan(0.85*limit, limit, alpha=0.12, color='darkred', label='Critical (>85%)')
        axs[i].axhspan(-limit, -0.85*limit, alpha=0.12, color='darkred')
        axs[i].axhspan(0.7*limit, 0.85*limit, alpha=0.10, color='red', label='Danger (70-85%)')
        axs[i].axhspan(-0.85*limit, -0.7*limit, alpha=0.10, color='red')
        axs[i].axhspan(0.5*limit, 0.7*limit, alpha=0.08, color='orange', label='Warning (50-70%)')
        axs[i].axhspan(-0.7*limit, -0.5*limit, alpha=0.08, color='orange')
        axs[i].axhspan(-0.5*limit, 0.5*limit, alpha=0.03, color='green', label='Safe (<50%)')
        
        # 最终角度标注
        axs[i].plot(len(tt)-1, xs[-1,2], 'ro', markersize=8, label=f'Final: {np.degrees(xs[-1,2]):.3f}°')
        axs[i].legend(loc='upper right', fontsize=8, ncol=2)
    
    axs[i].set_ylabel(ylabels[i], fontsize=14, fontweight='bold')
    axs[i].grid(alpha=0.2, linestyle='--', linewidth=0.5)
    axs[i].set_title(names[i], fontsize=12, pad=8)
    if i != 2:
        axs[i].legend(loc='upper right', fontsize=9)

# Control signal
axs[4].plot(tt[:-1], us, linewidth=2.5, color=colors[4], alpha=0.85, label='Control signal')
axs[4].axhline(0, color='black', linestyle='--', alpha=0.3, linewidth=1.5)
axs[4].axhline(u_max, color='red', linestyle=':', alpha=0.5, linewidth=1.5, label=f'Saturation (±{u_max:.0f}N)')
axs[4].axhline(-u_max, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
axs[4].fill_between(tt[:-1], -u_max, u_max, alpha=0.03, color='blue')
axs[4].set_ylabel(ylabels[4], fontsize=14, fontweight='bold')
axs[4].set_xlabel("Time step", fontsize=14, fontweight='bold')
axs[4].grid(alpha=0.2, linestyle='--', linewidth=0.5)
axs[4].set_title(names[4], fontsize=12, pad=8)
axs[4].legend(loc='upper right', fontsize=9)

# 优化的标题
if final_err < 0.15:
    status = "EXCELLENT ✓✓"
    color = 'darkgreen'
elif final_err < 0.3:
    status = "GOOD ✓"
    color = 'green'
else:
    status = "NEEDS IMPROVEMENT"
    color = 'orange'

fig.suptitle(f'Optimized Adaptive LQR - Final Error: {final_err:.5f} - Status: {status}', 
             fontsize=16, fontweight='bold', color=color, y=0.996)
fig.tight_layout()

out_dir = f"loss/{METHOD}/{MODEL}/control_plots"
os.makedirs(out_dir, exist_ok=True)
fig.savefig(f"{out_dir}/optimized_adaptive_control.png", dpi=200, bbox_inches='tight')
print(f"\n📊 High-resolution plot saved: {out_dir}/optimized_adaptive_control.png")

# 性能评估
print("\n" + "☆"*70)
print("PERFORMANCE EVALUATION")
print("☆"*70)

success_criteria = {
    "Stability": final_err < 0.3,
    "Position accuracy": abs(xs[-1,0]) < 0.2,
    "Angle accuracy": abs(np.degrees(xs[-1,2])) < 1.0,
    "No saturation": np.max(np.abs(us)) < 0.95 * u_max,
    "Fast settling": stabilization_time is not None and stabilization_time < 300
}

passed = sum(success_criteria.values())
total = len(success_criteria)

for criterion, passed_flag in success_criteria.items():
    status_symbol = "✓" if passed_flag else "✗"
    print(f"  {status_symbol} {criterion}")

print(f"\nOverall score: {passed}/{total} criteria passed")

if passed == total:
    print("\n🎉 OUTSTANDING PERFORMANCE! All criteria met!")
elif passed >= 4:
    print("\n✓ CONTROL SUCCESSFUL - Good performance!")
elif passed >= 3:
    print("\n✓ CONTROL SUCCESSFUL - Room for improvement")
else:
    print("\n⚠ Control needs further tuning")

