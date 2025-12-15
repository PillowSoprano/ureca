import numpy as np
args = {
'batch_size': 256,
'import_saved_data': False,
'val_frac': 0.2,
'lr1': 1e-4,  # Reduced from 3e-4 for numerical stability
'gamma': 0.9,
'num_epochs' : 401,
'weight_decay': 0.001,
'state_dim': 145,  # 实际废水系统状态维度（从ss_open.txt）
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
},
'waste_water':
{
    'pred_horizon': 20,      # 预测时域（污水处理系统动态较慢）
    'control_horizon': 20,    # 控制时域
    'old_horizon': 20,        # 历史观测时域
    'latent_dim': 64,         # 潜变量维度
    'd_conv': 10,             # Mamba卷积核尺寸
    'delta': 10,              # 离散化步长
    'total_data_size': 10000, # 训练数据总量
    'total_data_size_test': 2000,  # 测试数据量
    'max_ep_steps': 1000,     # 最大episode步数
    'max_ep_steps_test': 500, # 测试最大步数
    'optimize_step': 50,      # 优化步长
    'loop_test': 500,         # 测试循环次数
    'hidden_dim': 256,        # 隐层维度
    'h_dim': 256,             # GRU隐层维度（用于KoVAE）
    'z_dim': 64,              # KoVAE潜变量维度
    'disturbance': 0,         # 扰动水平
    'mamba':{
        # MPC权重矩阵（针对145维状态，4维动作）
        # 这里使用简化的对角矩阵，实际应根据具体状态变量调整
        'Q': np.eye(145) * 1.0,      # 状态权重（145x145）
        'R': np.eye(4) * 0.1,         # 控制输入权重（4x4）
        'P': np.eye(145) * 10.0,      # 终端状态权重（145x145）
    },
    'DKO':{
        'Q': np.eye(145) * 1.0,
        'R': np.eye(4) * 0.1,
        'P': np.eye(145) * 10.0,
    },
    'MLP':{
        'Q': np.eye(145) * 1.0,
        'R': np.eye(4) * 0.1,
        'P': np.eye(145) * 5.0,
    },
}
}

