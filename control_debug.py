# control_debug.py
# yongyu诊断环境问题
import os, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import args_new as new_args

METHOD = "kovae"
MODEL  = "cartpole"

args = dict(new_args.args, **new_args.ENV_PARAMS[MODEL])
args["env"] = MODEL

# 加载环境
if MODEL == "cartpole":
    from envs.cartpole import CartPoleEnv_adv as dreamer
elif MODEL == "cartpole_V":
    from envs.cartpole_V import CartPoleEnv_adv as dreamer
else:
    raise ValueError("only cartpole variants")

env = dreamer().unwrapped
u_max = float(env.action_space.high[0])

print("="*70)
print("ENVIRONMENT DIAGNOSTICS")
print("="*70)
print(f"Environment: {MODEL}")
print(f"Action space: {env.action_space}")
print(f"u_max = {u_max}")
print(f"Observation space: {env.observation_space}")

# 检查环境的终止条件
if hasattr(env, 'x_threshold'):
    print(f"Position threshold: ±{env.x_threshold}")
if hasattr(env, 'theta_threshold_radians'):
    print(f"Angle threshold: ±{env.theta_threshold_radians} rad (±{np.degrees(env.theta_threshold_radians):.1f}°)")

print("\n" + "="*70)
print("TEST 1: Zero Control (Natural Dynamics)")
print("="*70)

# 零控制看环境自然演化
rst = env.reset()
obs = rst[0] if isinstance(rst, tuple) else rst
print(f"Initial state: {obs}")

xs_zero, us_zero = [obs.copy()], []
for t in range(100):
    u = np.array([0.0])  # ！！！零控制
    step_out = env.step(u)
    if isinstance(step_out, tuple) and len(step_out) == 5:
        obs, r, terminated, truncated, info = step_out
        done = terminated or truncated
    else:
        obs, r, done, info = step_out
    
    xs_zero.append(obs.copy())
    us_zero.append(0.0)
    
    if done:
        print(f"Episode ended at step {t}")
        print(f"Final state: {obs}")
        print(f"Reason: terminated={terminated if 'terminated' in locals() else 'N/A'}, truncated={truncated if 'truncated' in locals() else 'N/A'}")
        break

xs_zero = np.array(xs_zero)
print(f"Survived {len(xs_zero)} steps with zero control")

print("\n" + "="*70)
print("TEST 2: Simple Proportional Control")
print("="*70)

# 简单比例控制
rst = env.reset()
obs = rst[0] if isinstance(rst, tuple) else rst
print(f"Initial state: {obs}")

X_REF = np.zeros(4)
xs_prop, us_prop = [obs.copy()], []

# 这是非常保守的增益
Kp = np.array([1.0, 0.5, 10.0, 2.0])

for t in range(200):
    err = obs - X_REF
    u = -Kp @ err
    u = np.clip(u, -u_max, u_max)
    
    step_out = env.step(np.array([u]))
    if isinstance(step_out, tuple) and len(step_out) == 5:
        obs, r, terminated, truncated, info = step_out
        done = terminated or truncated
    else:
        obs, r, done, info = step_out
    
    xs_prop.append(obs.copy())
    us_prop.append(u)
    
    if t % 20 == 0:
        print(f"Step {t:3d}: pos={obs[0]:6.3f}, angle={obs[2]:7.4f}, u={u:7.3f}")
    
    if done:
        print(f"\nEpisode ended at step {t}")
        print(f"Final state: pos={obs[0]:.4f}, vel={obs[1]:.4f}, angle={obs[2]:.4f}, ang_vel={obs[3]:.4f}")
        if hasattr(env, 'x_threshold') and abs(obs[0]) > env.x_threshold:
            print(f"FAIL: Position limit exceeded (|{obs[0]:.3f}| > {env.x_threshold})")
        if hasattr(env, 'theta_threshold_radians') and abs(obs[2]) > env.theta_threshold_radians:
            print(f"FAIL: Angle limit exceeded (|{obs[2]:.3f}| > {env.theta_threshold_radians:.3f})")
        break

xs_prop = np.array(xs_prop)
us_prop = np.array(us_prop)
print(f"Survived {len(xs_prop)} steps with proportional control")

print("\n" + "="*70)
print("TEST 3: Stronger LQR Control")
print("="*70)

# 更强的 LQR
rst = env.reset()
obs = rst[0] if isinstance(rst, tuple) else rst
print(f"Initial state: {obs}")

# 根据环境动力学调整的 LQR 增益
# 对于标准倒立摆，典型值：K = [-10, -15, 50, 15]
K_lqr = np.array([-5.0, -8.0, 40.0, 10.0])

xs_lqr, us_lqr = [obs.copy()], []

for t in range(300):
    err = obs - X_REF
    u = -K_lqr @ err
    u = np.clip(u, -u_max, u_max)
    
    step_out = env.step(np.array([u]))
    if isinstance(step_out, tuple) and len(step_out) == 5:
        obs, r, terminated, truncated, info = step_out
        done = terminated or truncated
    else:
        obs, r, done, info = step_out
    
    xs_lqr.append(obs.copy())
    us_lqr.append(u)
    
    if t % 30 == 0:
        err_norm = np.linalg.norm(obs - X_REF)
        print(f"Step {t:3d}: pos={obs[0]:6.3f}, angle={obs[2]:7.4f}, err={err_norm:.4f}, u={u:7.3f}")
    
    if done:
        print(f"\nEpisode ended at step {t}")
        print(f"Final state: {obs}")
        break
    
    # 检查稳定性
    if t > 100:
        recent_err = np.linalg.norm(xs_lqr[-50:] - X_REF, axis=1)
        if np.mean(recent_err) < 0.1:
            print(f"\nStabilized at step {t}!")
            break

xs_lqr = np.array(xs_lqr)
us_lqr = np.array(us_lqr)
print(f"Survived {len(xs_lqr)} steps with LQR control")

# 可视化对比
fig, axs = plt.subplots(5, 3, figsize=(16, 14), sharex='col')
fig.suptitle('Control Comparison: Zero / Proportional / LQR', fontsize=14, fontweight='bold')

datasets = [
    ('Zero Control', xs_zero, us_zero),
    ('Proportional', xs_prop, us_prop),
    ('LQR', xs_lqr, us_lqr)
]

state_names = ['Position', 'Velocity', 'Angle', 'Angular Vel', 'Control']
colors = ['steelblue', 'mediumseagreen', 'coral', 'mediumpurple', 'forestgreen']

for col, (title, xs, us) in enumerate(datasets):
    tt = np.arange(len(xs))
    
    for row in range(4):
        axs[row, col].plot(tt, xs[:, row], color=colors[row], linewidth=1.5)
        axs[row, col].axhline(0, color='red', linestyle='--', alpha=0.5)
        axs[row, col].set_ylabel(state_names[row] if col == 0 else '')
        axs[row, col].grid(alpha=0.3)
        
        if row == 0:
            axs[row, col].set_title(title, fontweight='bold')
        
        # 标记终止条件
        if hasattr(env, 'x_threshold') and row == 0:
            axs[row, col].axhline(env.x_threshold, color='orange', linestyle=':', alpha=0.6)
            axs[row, col].axhline(-env.x_threshold, color='orange', linestyle=':', alpha=0.6)
        if hasattr(env, 'theta_threshold_radians') and row == 2:
            axs[row, col].axhline(env.theta_threshold_radians, color='orange', linestyle=':', alpha=0.6)
            axs[row, col].axhline(-env.theta_threshold_radians, color='orange', linestyle=':', alpha=0.6)
    
    # 控制量
    tt_u = np.arange(len(us))
    axs[4, col].plot(tt_u, us, color=colors[4], linewidth=1.5)
    axs[4, col].axhline(0, color='gray', linestyle=':', alpha=0.5)
    axs[4, col].axhline(u_max, color='red', linestyle=':', alpha=0.4)
    axs[4, col].axhline(-u_max, color='red', linestyle=':', alpha=0.4)
    axs[4, col].set_ylabel('Control' if col == 0 else '')
    axs[4, col].set_xlabel('Time step')
    axs[4, col].grid(alpha=0.3)

fig.tight_layout()
out_dir = f"loss/{METHOD}/{MODEL}/control_plots"
os.makedirs(out_dir, exist_ok=True)
fig.savefig(f"{out_dir}/diagnostic_comparison.png", dpi=150)
print(f"\nDiagnostic plot saved: {out_dir}/diagnostic_comparison.png")

# 结论
print("\n" + ":D"*70)
print("DIAGNOSTIC SUMMARY")
print("( ´ ▽ ` )ﾉ"*70)

if len(xs_lqr) > 200:
    print("✓ LQR control works! The environment is controllable.")
    print("\nRecommendation: Use LQR gains K = [-5, -8, 40, 10]")
    print("Your original control gains were too weak.")
elif len(xs_prop) > 100:
    print("✓ Proportional control works but LQR failed.")
    print("\nRecommendation: Tune LQR gains or check control saturation.")
elif len(xs_zero) > 50:
    print("⚠ System is open-loop stable but closed-loop unstable.")
    print("\nPossible issues:")
    print("- Control gains create instability")
    print("- Control delay or discretization errors")
    print("- Wrong control sign")
else:
    print("✗ System fails even with zero control!")
    print("\nPossible issues:")
    print("- Environment has very tight termination conditions")
    print("- Initial conditions are outside stable region")
    print("- Environment implementation has bugs")

print("\nTo fix your control:")
print("1. Check env termination conditions (x_threshold, theta_threshold)")
print("2. Increase control gains gradually")
print("3. Add integral action for steady-state error")
print("4. Verify control sign matches dynamics")
print("(^з^)-☆"*70)

