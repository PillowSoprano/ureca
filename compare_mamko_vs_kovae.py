# compare_mamko_vs_kovae.py
import os, json, numpy as np, torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import args_new as new_args
from replay_fouling import ReplayMemory
from robustness_eval import evaluation   # uses your built-ins

plt.rcParams["figure.dpi"] = 160

MODEL     = "cartpole"     # <-- set your env/model name
METHODS   = ["mamba", "kovae"]  # MamKO vs baseline (rename if needed)
RESULTDIR = f"results/{MODEL}/compare"
os.makedirs(RESULTDIR, exist_ok=True)

# ---------- helpers ----------
def patch_io_paths(args, tag):
    """Ensure shift/scale & model paths exist even if args lacks them."""
    base = args.get('save_dir', f'./log/{MODEL}/{tag}')
    os.makedirs(base, exist_ok=True)
    args.setdefault('save_model_path', os.path.join(base, 'model.pt'))
    args.setdefault('save_opti_path',  os.path.join(base, 'opti.pt'))
    args.setdefault('shift_x',         os.path.join(base, 'shift_x.txt'))
    args.setdefault('scale_x',         os.path.join(base, 'scale_x.txt'))
    args.setdefault('shift_u',         os.path.join(base, 'shift_u.txt'))
    args.setdefault('scale_u',         os.path.join(base, 'scale_u.txt'))
    return args

def build_args(method_name):
    args = dict(new_args.args)                      # base hyperparams
    args.update(new_args.ENV_PARAMS[MODEL])         # env-specific
    args['method'] = method_name
    # bounds for controller may be needed by robustness_eval/controller
    env = get_env_from_name({'env_name': MODEL})
    args['s_bound_low']  = env.observation_space.low
    args['s_bound_high'] = env.observation_space.high
    args['a_bound_low']  = env.action_space.low
    args['a_bound_high'] = env.action_space.high
    args['reference']    = getattr(env, 'xs', None)
    patch_io_paths(args, method_name)
    return args, env

def load_model(method_name, args):
    if method_name == 'mamba':
        from MamKO import Koopman_Desko as Model
    elif method_name == 'kovae':
        # TODO: adjust to your baseline's class
        from DKO import Koopman_Desko as Model   # <- example; replace if you have a real KOVAE module
    else:
        raise ValueError(f'Unknown method {method_name}')
    m = Model(args)
    # best effort restore (common in your codebase)
    if hasattr(m, 'parameter_restore'):
        m.parameter_restore(args)
    elif hasattr(m, 'restore'):
        m.restore()
    return m

def get_shift_scale(args):
    sx = np.loadtxt(args['shift_x']).astype(np.float32)
    scx = np.loadtxt(args['scale_x']).astype(np.float32)
    su = np.loadtxt(args['shift_u']).astype(np.float32)
    scu = np.loadtxt(args['scale_u']).astype(np.float32)
    scx[scx == 0] = 1.0
    scu[scu == 0] = 1.0
    return sx, scx, su, scu

def normalize(x, shift, scale):
    return (x - shift) / np.clip(scale, 1e-6, None)

def denormalize(xn, shift, scale):
    return xn * np.clip(scale, 1e-6, None) + shift

# ---------- modeling performance ----------
@torch.no_grad()
def eval_modeling_mse(model, args, split='val', nstep=1, batch_size=256):
    data = ReplayMemory(args)   # will populate dataset_train/val/test
    if split == 'train':
        ds = data.dataset_train
    elif split == 'val':
        ds = data.dataset_val
    else:
        ds = data.dataset_test
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    # Figure out how to call 1-step prediction.
    # Try common patterns found in MamKO/DKO:
    # (x_t, u_t) -> predict x_{t+1} (or delta)
    has_forward = hasattr(model, 'forward')
    has_predict = hasattr(model, 'predict')
    has_step    = hasattr(model, 'step')

    mses = []
    for xb, yb in loader:
        # xb shape [B, state_dim (+ action?)], yb = next state (dataset uses x_choice, y)
        x_np = xb.numpy().astype(np.float32)
        y_np = yb.numpy().astype(np.float32)

        # Best-effort: call model to get x_{t+1} prediction from x_t (and u_t if required).
        if has_predict:
            y_hat = model.predict(x_np)
        elif has_step:
            y_hat = model.step(x_np)
        elif has_forward:
            # many models take torch tensor
            y_hat = model.forward(torch.from_numpy(x_np)).detach().cpu().numpy()
        else:
            raise RuntimeError("Model has no forward/predict/step for 1-step eval")

        err = ((y_hat - y_np) ** 2).mean(axis=1)   # per-sample MSE
        mses.append(err)

    mses = np.concatenate(mses, axis=0)
    return float(mses.mean()), mses

def plot_modeling_log(method2mse, split, outpath):
    methods = list(method2mse.keys())
    vals = [method2mse[m][0] for m in methods]
    plt.figure()
    plt.yscale('log')
    plt.bar(methods, vals)
    plt.ylabel(f'{split} MSE (log scale)')
    plt.title(f'Modeling {split} MSE')
    for i, v in enumerate(vals):
        plt.text(i, v, f'{v:.3e}', ha='center', va='bottom', fontsize=8, rotation=90)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# ---------- control performance ----------
def eval_control_cost(model, env, method_name):
    """
    Uses your robustness_eval.evaluation to roll out and return a scalar cost
    (falls back to sum of path costs if needed).
    """
    VAR = dict(
        env_name=MODEL,
        eval_render=False,
        evaluation_form='dynamic',
        max_ep_steps_test=env._max_episode_steps if hasattr(env, "_max_episode_steps") else 200,
        log_path=os.path.join('log', MODEL, method_name),
        loc='compare',
        iter=0,
        form='default',
        eval_list=[]
    )
    diag, _ = evaluation(VAR, env, model, verbose=False)  # policy==model for your controllers
    # Try common keys:
    for k in ['cost', 'total_cost', 'J', 'cum_cost', 'tracking_mse']:
        if k in diag:
            return float(diag[k]), diag
    # Fallback: look for any numeric and pick the most “cost-like”
    numeric_items = {k: v for k, v in diag.items() if isinstance(v, (int, float, np.floating))}
    if numeric_items:
        # heuristic: prefer larger-is-worse
        return float(max(numeric_items.values())), diag
    # if everything fails, return NaN
    return float('nan'), diag

def plot_control_log(costs, outpath):
    methods = list(costs.keys())
    vals = [costs[m][0] for m in methods]
    plt.figure()
    plt.yscale('log')
    plt.bar(methods, vals)
    plt.ylabel('Control cost J (log scale)')
    plt.title('Closed-loop performance')
    for i, v in enumerate(vals):
        if np.isfinite(v):
            plt.text(i, v, f'{v:.3e}', ha='center', va='bottom', fontsize=8, rotation=90)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# ---------- main ----------
def main():
    modeling_summary = {'train':{}, 'val':{}, 'test':{}}
    control_summary  = {}
    env_cache = None

    for m in METHODS:
        args, env = build_args(m)
        env_cache = env
        model = load_model(m, args)

        # modeling
        for split in ['train','val','test']:
            mse_mean, _ = eval_modeling_mse(model, args, split=split)
            modeling_summary[split][m] = (mse_mean, )

        # control
        J, diag = eval_control_cost(model, env, m)
        control_summary[m] = (J, diag)

    # plots (log space)
    for split in ['train','val','test']:
        plot_modeling_log(modeling_summary[split], split,
                          os.path.join(RESULTDIR, f'modeling_{split}_log.png'))
    plot_control_log(control_summary,
                     os.path.join(RESULTDIR, f'control_cost_log.png'))

    # print a tiny table
    print("\nModeling MSE (mean):")
    for split in ['train','val','test']:
        for m in METHODS:
            print(f"{split:>5} | {m:>6}: {modeling_summary[split][m][0]:.3e}")
    print("\nControl cost J:")
    for m in METHODS:
        print(f"{m:>6}: {control_summary[m][0]:.3e}")

if __name__ == "__main__":
    main()
