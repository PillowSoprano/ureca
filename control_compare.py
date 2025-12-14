# control_compare.py
import os, numpy as np, cvxpy as cp, torch, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import args_new as new_args
from replay_fouling import ReplayMemory
from mbr_actions import clip_mbr_action, summarize_actuators

plt.rcParams["figure.dpi"]=150

METHODS = ["kovae","mamba"]   # compare these two
MODEL   = "cartpole"
H       = 40
T_SIM   = 200
THETA_TOL   = 0.01   # rad
THETAD_TOL  = 0.02   # rad/s

def get_args(method):
    args = dict(new_args.args, **new_args.ENV_PARAMS[MODEL])
    args["env"] = MODEL
    args["method"] = method
    args["control"] = True
    fold = f"save_model/{method}/{MODEL}"
    os.makedirs(fold, exist_ok=True)
    args["save_model_path"] = f"{fold}/model.pt"
    args["save_opti_path"]  = f"{fold}/opti.pt"
    args["shift_x"]         = f"{fold}/shift_x.txt"
    args["scale_x"]         = f"{fold}/scale_x.txt"
    args["shift_u"]         = f"{fold}/shift_u.txt"
    args["scale_u"]         = f"{fold}/scale_u.txt"
    return args

def load_model(method, args):
    if method == "kovae":
        from kovae_model import Koopman_Desko
    elif method == "mamba":
        from MamKO import Koopman_Desko
    else:
        raise ValueError(f"Unknown method {method}")
    m = Koopman_Desko(args)
    # 统一用参数恢复接口
    if hasattr(m, "parameter_restore"):
        m.parameter_restore(args)
    elif hasattr(m, "restore"):
        m.restore()
    return m

def norm_fns(args):
    shift_x = np.loadtxt(args["shift_x"]).astype(np.float32)
    scale_x = np.loadtxt(args["scale_x"]).astype(np.float32)
    def norm_x(x):   return (x - shift_x) / np.clip(scale_x, 1e-6, None)
    def denorm_x(x): return x * np.clip(scale_x, 1e-6, None) + shift_x
    return norm_x, denorm_x

def encode_x(policy, x_np, use_action=False, act_dim=1):
# 编码当前状态到 latent；若无 encoder，则直接使用归一化状态作为 latent
    xn = policy._norm_x(x_np.astype(np.float32))
    x = torch.from_numpy(xn[None, None, :]).float().to(policy.device)
    if use_action:
        u0 = torch.zeros((1, 1, act_dim), device=policy.device)
        xin = torch.cat([x, u0], dim=-1)
    else:
        xin = x

    if hasattr(policy, "model") and hasattr(policy.model, "enc"):
        policy.model.eval()
        with torch.no_grad():
            z, _ = policy.model.enc(xin)
        return z.squeeze().detach().cpu().numpy()
    else:
        # 没有 enc：使用归一化状态作为 latent
        return xn.astype(np.float32)

def fit_abcd(policy, args, env):
# 从训练数据拟合 z_{t+1} = A z_t + B u_t + c 以及 x ≈ C z + d
    rp = ReplayMemory(args, env, predict_evolution=True)
    loader = DataLoader(rp.dataset_train, batch_size=128, shuffle=True)
    Z0_list, Z1_list, U_list, X0_list = [], [], [], []
    dev = policy.device

    with torch.no_grad():
        for x, u in loader:
            x = x.float().to(dev)
            u = u.float().to(dev)
            if u.dim() == 2:
                u = u[:, None, :].expand(x.size(0), x.size(1), -1)
            xin = torch.cat([x, u], dim=-1) if args.get("use_action", False) else x

            if hasattr(policy, "model") and hasattr(policy.model, "enc"):
                policy.model.eval()
                z, _ = policy.model.enc(xin)  # [B,T,k]
            else:
                z = x  # [B,T,Dx]：无 encoder 时，直接用状态作为 latent!!

            Tm = min(z.size(1), u.size(1))
            if Tm < 2:
                continue

            z0 = z[:, :Tm-1, :].reshape(-1, z.size(-1)).cpu().numpy()
            z1 = z[:, 1:Tm,  :].reshape(-1, z.size(-1)).cpu().numpy()
            uu = u[:, :Tm-1, :].reshape(-1, u.size(-1)).cpu().numpy()
            x0 = x[:, :Tm-1, :].reshape(-1, x.size(-1)).cpu().numpy()

            Z0_list.append(z0); Z1_list.append(z1); U_list.append(uu); X0_list.append(x0)
            if sum(t.shape[0] for t in Z0_list) > 200_000:
                break

    Z0 = np.concatenate(Z0_list, 0)
    Z1 = np.concatenate(Z1_list, 0)
    U  = np.concatenate(U_list , 0)
    X0 = np.concatenate(X0_list, 0)  # normalized x

    N  = min(Z0.shape[0], Z1.shape[0], U.shape[0], X0.shape[0])
    Z0, Z1, U, X0 = Z0[:N], Z1[:N], U[:N], X0[:N]

    # z_{t+1} = A z_t + B u_t + c
    Phi = np.concatenate([Z0, U, np.ones((N, 1))], 1)               # [N,k+m+1]
    Theta, *_ = np.linalg.lstsq(Phi, Z1, rcond=None)                # [(k+m+1)×k]
    Theta = np.asarray(Theta)
    k = Z1.shape[1]
    m = U.shape[1]
    A = Theta[:k,     :].T
    B = Theta[k:k+m,  :].T
    c = Theta[-1,     :]

    # decoder: x_norm ≈ C_norm z + d_norm
    Phi_z = np.concatenate([Z0, np.ones((N, 1))], 1)
    W_x, *_ = np.linalg.lstsq(Phi_z, X0, rcond=None)                # [(k+1)×Dx]
    Cn = W_x[:-1, :].T
    dn = W_x[-1,  :].T

    # map decoder to physical units
    shift_x = np.loadtxt(args["shift_x"]).astype(np.float32)
    scale_x = np.loadtxt(args["scale_x"]).astype(np.float32)
    S = np.diag(np.clip(scale_x, 1e-6, None))
    C = (S @ Cn).astype(np.float32)         # [Dx×k]
    d = (S @ dn + shift_x).astype(np.float32)

    # sign correction per dimension（保证每个维度的相关性为正!）
    X0_phys = (X0 * scale_x) + shift_x
    Xhat0   = (C @ Z0.T + d[:, None]).T
    signs = []
    for i in range(X0.shape[1]):
        xi, xhi = X0_phys[:, i], Xhat0[:, i]
        if np.std(xi) < 1e-6 or np.std(xhi) < 1e-6:
            s = 1.0
        else:
            ccf = np.corrcoef(xi, xhi)[0, 1]
            s = 1.0 if np.isnan(ccf) else (1.0 if ccf >= 0 else -1.0)
        signs.append(s)
    Sgn  = np.diag(np.array(signs, dtype=np.float32))
    C_adj = (Sgn @ C).astype(np.float32)
    d_adj = (Sgn @ d).astype(np.float32)

    return A.astype(np.float32), B.astype(np.float32), c.astype(np.float32), C_adj, d_adj

def build_policy(method):
    args = get_args(method)

    # env
    from envs.cartpole import CartPoleEnv_adv as dreamer
    env = dreamer().unwrapped
    action_low = getattr(env, "action_low", env.action_space.low)
    action_high = getattr(env, "action_high", env.action_space.high)
    args["state_dim"] = env.observation_space.shape[0]
    args["act_dim"]   = env.action_space.shape[0]
    u_max = np.asarray(action_high, dtype=float)

    # model
    model = load_model(method, args)

    # set device: prefer model.device, else mps -> cuda -> cpu
    if hasattr(model, "device"):
        device = model.device
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.device = device

    # MamKO 兼容：没有 .model 但有 .net
    if not hasattr(model, "model") and hasattr(model, "net"):
        model.model = model.net

    # 只有存在 .model 才 to(device)
    if hasattr(model, "model"):
        try:
            model.model.to(device)
        except Exception:
            pass

    # 归一化函数句柄
    nx, _ = norm_fns(args)
    model._norm_x = nx

    # 拟合线性模型与解码器
    A, B, c, C, d = fit_abcd(model, args, env)

    # 输入方向（角度维=2）
    J = C @ B
    SIGN_U = float(np.sign(J[2, 0]) or 1.0)

    def clip_action(u_vec):
        return clip_mbr_action(u_vec, action_low, action_high) if u_vec.shape[-1] == 4 else np.clip(u_vec, -u_max, u_max)

    # 代价矩阵（只罚 theta/thetadot）
    Q = new_args.ENV_PARAMS[MODEL]["MLP"]["Q"].astype(np.float32) * 0.0
    Q[2, 2] = 50.0
    Q[3, 3] = 5.0
    R = (1.5 * new_args.ENV_PARAMS[MODEL]["MLP"]["R"]).astype(np.float32)
    term_w = 120.0

    # 稳态点 (z_ss,u_ss)，目标 X_REF=0
    X_REF = np.zeros(args["state_dim"], dtype=np.float32)
    k = A.shape[0]; m = B.shape[1]
    Zss = cp.Variable(k)
    Uss = cp.Variable(m)
    prob = cp.Problem(
        cp.Minimize(cp.sum_squares(Zss) + 1e-3 * cp.sum_squares(Uss)),
        [Zss == A @ Zss + (B * SIGN_U) @ Uss + c,
         C @ Zss + d == X_REF]
    )
    prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=8000)
    z_ss = (Zss.value if Zss.value is not None else np.zeros(k)).astype(np.float32)
    u_ss = (Uss.value if Uss.value is not None else np.zeros(m)).astype(np.float32)

    # MPC 控制器
    def mpc_control(x_np):
        z0 = encode_x(model, x_np, use_action=False, act_dim=args["act_dim"])
        Zdev = cp.Variable((k, H + 1))
        Udev = cp.Variable((m, H))
        cost = 0
        cons = [Zdev[:, 0] == z0 - z_ss]
        for t in range(H):
            x_pred = C @ (Zdev[:, t] + z_ss) + d
            x_err  = x_pred - X_REF
            cost  += cp.quad_form(x_err, Q) + cp.quad_form(Udev[:, t], R)
            cons  += [Zdev[:, t + 1] == A @ Zdev[:, t] + (B * SIGN_U) @ Udev[:, t]]
            cons  += [cp.abs(Udev[:, t] + u_ss) <= u_max]
        xH = C @ (Zdev[:, H] + z_ss) + d
        cost += term_w * cp.quad_form(xH - X_REF, Q)
        cp.Problem(cp.Minimize(cost), cons).solve(
            solver=cp.OSQP, verbose=False, eps_abs=1e-4, eps_rel=1e-4, max_iter=8000
        )
        if Udev.value is None:
            return np.zeros((m,), dtype=float)
        u0 = (Udev.value[:, 0] + u_ss) * SIGN_U
        return clip_action(u0)

    return env, mpc_control, action_low, action_high, u_max

def simulate(env, policy, T=T_SIM):
    rst = env.reset()
    obs = rst[0] if isinstance(rst, tuple) else rst
    xs, us = [], []
    for t in range(T):
        u = clip_action(np.array(policy(obs), dtype=float).reshape(-1))
        step = env.step(u)
        if len(step) == 5:
            obs, r, terminated, truncated, info = step
            done = bool(terminated) or bool(truncated)
        else:
            obs, r, done, info = step
        xs.append(np.array(obs, float))
        us.append(np.array(u, float))
        if done:
            break
    return np.array(xs), np.array(us)

def metrics(xs, us, u_max, action_low=None, action_high=None):
    T = xs.shape[0]
    theta = xs[:, 2]
    thetad = xs[:, 3]
    ok = (np.abs(theta) < THETA_TOL) & (np.abs(thetad) < THETAD_TOL)
    settle = None
    for t in range(T):
        if ok[t] and ok[t:].all():
            settle = t
            break
    last = slice(max(0, T - 50), T)
    ss_abs = np.mean(np.abs(xs[last, :]), axis=0)    # steady-state |x| ave
    if us.ndim == 2 and us.shape[1] == 4 and action_low is not None and action_high is not None:
        util = summarize_actuators(us, action_low, action_high)
        u_mean = float(np.mean(np.abs(us)))
        sat = {k: v["saturation_pct"] for k, v in util.items()}
        return {
            "settling_step": -1 if settle is None else int(settle),
            "ss_abs_x": ss_abs.tolist(),
            "u_mean_abs": u_mean,
            "u_sat_percent": sat,
        }
    u_sat = np.mean(np.abs(us) >= (0.999 * u_max)) * 100.0
    u_mean = float(np.mean(np.abs(us)))
    return {
        "settling_step": -1 if settle is None else int(settle),
        "ss_abs_x": ss_abs.tolist(),
        "u_mean_abs": u_mean,
        "u_sat_percent": float(u_sat),
    }

def main():
    outdir = f"loss/compare/{MODEL}"
    os.makedirs(outdir, exist_ok=True)

    rows = []
    for method in METHODS:
        print("===", method, "===")
        env, ctrl, action_low, action_high, u_max = build_policy(method)
        xs, us = simulate(env, ctrl, T_SIM)

        # plot (time-series，保持线性坐标更直观；如需 log，可对单维加 ax.set_yscale('log'))
        tt = np.arange(xs.shape[0])
        fig, axs = plt.subplots(xs.shape[1] + 1, 1, figsize=(8, 2.0 * (xs.shape[1] + 1)), sharex=True)
        for i in range(xs.shape[1]):
            axs[i].plot(tt, xs[:, i]); axs[i].set_ylabel(f"x[{i}]"); axs[i].grid(alpha=0.3)
        axs[-1].plot(tt, us); axs[-1].set_ylabel("u"); axs[-1].set_xlabel("t"); axs[-1].grid(alpha=0.3)
        fig.tight_layout()
        pth = f"{outdir}/control_{method}.png"
        fig.savefig(pth, dpi=160); plt.close(fig)
        print("saved", pth)

        # metrics
        m = metrics(xs, us, u_max, action_low=action_low, action_high=action_high)
        rows.append((method, m))

    # write metrics
    with open(f"{outdir}/metrics_control.txt", "w") as f:
        for method, m in rows:
            if isinstance(m["u_sat_percent"], dict):
                sat_str = ", ".join(f"{k}:{v:.2f}%" for k, v in m["u_sat_percent"].items())
            else:
                sat_str = f"{m['u_sat_percent']:.2f}%"
            f.write(f"{method}: settling_step={m['settling_step']}, "
                    f"ss_abs_x={m['ss_abs_x']}, "
                    f"u_mean_abs={m['u_mean_abs']:.4f}, "
                    f"u_sat%={sat_str}\n")
    print("Saved:", f"{outdir}/metrics_control.txt")

if __name__ == "__main__":
    main()
    # plz don't fail
