import numpy as np
args = {
'batch_size': 256,
'import_saved_data': False,
'val_frac': 0.2,
'lr1': 3e-4,
'gamma': 0.9,
'num_epochs' : 401,
'weight_decay': 0.001,
'state_dim': 156,
'act_dim': 4,

'SAVE_TEST' : "/test.pt",
'SAVE_TRAIN' : "/train.pt",
'SAVE_VAL' : "/val.pt",
'SAVE_DRAW' : "/draw.pt",

'apply_state_constraints': False,
'apply_action_constraints': True,
}
# koVAE 需要的通用超参（新增）
# 映射学习率名称，兼容我已有的kovae_model.py!
args['learning_rate'] = args.get('lr1', 1e-3)

# KoVAE 模型结构/损失超参
args.setdefault('z_dim', 64)        # 潜变量维度（适配 156 维状态）
args.setdefault('h_dim', 256)       # GRU 隐层维度
args.setdefault('layers', 1)        # GRU 层数
args.setdefault('alpha', 0.1)       # posterior-prior 一致性权重
args.setdefault('beta', 1e-3)       # KL 权重（建议做退火）
# 避免与上面的折扣因子 gamma=0.9 冲突，选择用 eig_gamma
args.setdefault('eig_gamma', 0.0)   # 谱正则权重（需要才 >0）
args.setdefault('eig_target', '<=1')# 谱约束目标：'< =1' 或 1.0
args.setdefault('eig_margin', 0.0)  # 谱约束边际
args.setdefault('grad_clip', 5.0)   # 梯度裁剪（长序列更稳定）
args.setdefault('use_action', True) # 是否把动作拼进输入一起建模
args.setdefault('training_mode', 'standard')  # 'standard' or 'hybrid'
args.setdefault('sim_rollout_length', 50)
args.setdefault('sim_step_size', 1)
args.setdefault('sim_noise_scale', 0.0)
args.setdefault('sim_fouling_perturb', 0.0)
args.setdefault('sim_batches', 8)
args.setdefault('control_weight', 0.0)
args.setdefault('sim_weight', 0.0)
args.setdefault('dropout', 0.1)
args.setdefault('layer_norm', True)
args.setdefault('activation', 'tanh')
args.setdefault('lr_scheduler', 'cosine')
args.setdefault('scheduler_gamma', 0.8)
args.setdefault('scheduler_step', 50)
args.setdefault('scheduler_min_lr', 1e-5)
args.setdefault('grid_sweep', False)
args.setdefault('sweep_epochs', 5)
args.setdefault('sweep_batch_sizes', [128, 256])
args.setdefault('sweep_seq_lengths', [20, 40])
args.setdefault('sweep_latent_dims', [32, 64])

ENV_PARAMS = {
'cartpole':
{
    'pred_horizon': 30,
    'control_horizon':30,
    'old_horizon':30,
    'latent_dim': 8, 
    'd_conv' : 10,
    'delta': 10,
    'total_data_size': 40000-100,
    'total_data_size_test': 4000,
    'max_ep_steps': 20000+40,
    'max_ep_steps_test':1000,
    'optimize_step':50,
    'loop_test':1000,
    'hidden_dim':64,
    'h_dim': 64,     # = hidden_dim
    'z_dim': 16,     # = latent_dim
    'disturbance' : 0,
    'mamba':{
        'Q':np.diag([1,0.01,100,0.01]),
        'R':np.diag([0.5]),
        'P':np.diag([5000,0,0,0]),
    },
    'DKO':{
        'Q':np.diag([1,0.01,100,0.01]),
        'R':np.diag([0.5]),
        'P':np.diag([5000,0,0,0]),
    },
    'MLP':{
        'Q':np.diag([1,0.01,20,0.01]),
        'R':np.diag([0.5]),
        'P':np.diag([5000,0,0,0]),
    },
},
'cartpole_V':
{
    'pred_horizon': 30,
    'control_horizon':30,
    'old_horizon':30,
    'latent_dim': 8, 
    'd_conv' : 15,
    'delta': 5,
    'hidden_dim':64,
    'h_dim': 64,     # = hidden_dim
    'z_dim': 16,     # = latent_dim
    'import_saved_data': False,
    'total_data_size': 40000-100,
    'total_data_size_test': 4000,
    'max_ep_steps': 20000+40,
    'max_ep_steps_test':1000,
    'optimize_step':50,
    'loop_test':1000,
    'disturbance' : 0,
    'mamba':{
        'Q':np.diag([1,0.0001,100,0.0001]),
        'R':np.diag([0.1]),
        'P':np.diag([5000,0,0,0]),
    },
    'DKO':{
        'Q':np.diag([1,0.0001,100,0.0001]),
        'R':np.diag([0.1]),
        'P':np.diag([2000,0,0,0]), # TODO: TOO LARGE MAY FAIL
    },
    'MLP':{
        'Q':np.diag([1,0.0001,100,0.0001]),
        'R':np.diag([0.1]),
        'P':np.diag([2000,0,0,0]), # TODO: TOO LARGE MAY FAIL
    },
}
}

