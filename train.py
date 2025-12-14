from replay_fouling import ReplayMemory
# from variant import *

# from three_tanks import three_tank_system as dreamer
# from MBR import MBR as dreamer

from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import gym

import numpy as np
import torch
import os

import argparse
from sim_interface import MBRTrajectorySimulator


parparser = argparse.ArgumentParser()
parparser.add_argument('method',type=str)
parparser.add_argument('model',type=str)

condition = parparser.parse_args()


def main():     
    import args_new as new_args
    args = new_args.args

    if condition.model == 'cartpole':
        from envs.cartpole import CartPoleEnv_adv as dreamer
    if condition.model == 'cartpole_V':
        from envs.cartpole_V import CartPoleEnv_adv as dreamer
        
    if condition.model == 'half_cheetah':
        # from envs.half_cheetah_cost import HalfCheetahEnv_cost as dreamer
        dreamer = gym.make('HalfCheetah-v2')
    args['env'] = condition.model

    

    args = dict(args,**new_args.ENV_PARAMS[condition.model])


    if condition.method == 'mamba':
        from MamKO import Koopman_Desko
        args['method'] = 'mamba'
    if condition.method == 'DKO':
        from DKO import Koopman_Desko
        args['method'] = 'DKO'
    if condition.method == 'MLP' or condition.method == 'LSTM' or condition.method == 'TRANS':
        from MLP import Koopman_Desko
        args['method'] = 'MLP'
        args['structure'] = condition.method
        # args['optimize_step'] = 20
        # === 新增：KoVAE 方法 ===
    if condition.method == 'kovae':
        from kovae_model import Koopman_Desko
        args['method'] = 'kovae'
        # 可选：默认超参
        args.setdefault('z_dim', 16)
        args.setdefault('h_dim', 64)
        args.setdefault('alpha', 0.1)
        args.setdefault('beta', 1e-3)
        args.setdefault('gamma', 0.0)       # 需要谱约束再调 >0
        args.setdefault('grad_clip', 1.0)
        args.setdefault('weight_decay', 1e-4)
        # 是否把动作拼进输入
        args.setdefault('use_action', False)

    args['continue_training'] = True
    
    for i in range(10):
        env = dreamer()
        env = env.unwrapped
        args['state_dim'] = env.observation_space.shape[0]
        args['act_dim'] = env.action_space.shape[0]
        args['control'] =  False

        fold_path = 'save_model/'+condition.method+'/'+str(condition.model)
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)

        args['save_model_path'] = fold_path+'/model.pt'
        args['save_opti_path']  = fold_path+'/opti.pt'
        args['shift_x']         = fold_path+'/shift_x.txt'
        args['scale_x']         = fold_path+'/scale_x.txt'
        args['shift_u']         = fold_path+'/shift_u.txt'
        args['scale_u']         = fold_path+'/scale_u.txt'
        
        model = Koopman_Desko(args)
        args['times_training'] = i
        train(args,model,env,i)

        if not args['continue_training']:
            break


class HybridDataset(Dataset):
    """Wraps existing datasets to optionally attach simulator targets."""

    def __init__(self, base, sim_states=None, has_sim=True):
        self.base = base
        self.sim_states = sim_states
        self.has_sim = has_sim and sim_states is not None

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, u = self.base[idx]
        if self.has_sim:
            sim = self.sim_states[idx]
            mask = np.ones(1, dtype=np.float32)
        else:
            sim = np.zeros_like(x)
            mask = np.zeros(1, dtype=np.float32)
        return x, u, sim, mask


def build_hybrid_rollouts(args):
    """Generate simulated rollouts seeded from ss_open.txt for hybrid mode."""

    try:
        seed_state = np.loadtxt('ss_open.txt')
    except OSError:
        return None

    sim_batches = int(args.get('sim_batches', 4))
    rollout_len = int(args.get('sim_rollout_length', 50))
    step_size = int(args.get('sim_step_size', 1))
    sim = MBRTrajectorySimulator(step_size=step_size,
                                 noise_scale=args.get('sim_noise_scale', 0.0),
                                 fouling_perturb=args.get('sim_fouling_perturb', 0.0))

    actions = np.random.uniform(low=sim.env.action_low,
                                high=sim.env.action_high,
                                size=(sim_batches, rollout_len, 4))
    rollouts = sim.rollout(actions, start_state=seed_state)
    states = rollouts['states']
    actions = actions.astype(np.float32)

    # match ReplayMemory slicing strategy
    seq_O = int(args.get('old_horizon', 0))
    seq_pred = int(args.get('pred_horizon', rollout_len))
    x_slices = []
    u_slices = []
    for b in range(sim_batches):
        traj_x = states[b]
        traj_u = actions[b]
        add_x = np.repeat(traj_x[0:1].transpose(), seq_O, axis=1).T
        add_u = np.repeat(np.zeros_like(traj_u[0:1]).transpose(), seq_O, axis=1).T
        traj_x = np.concatenate((add_x, traj_x))
        traj_u = np.concatenate((add_u, traj_u))
        j = seq_O
        while j + seq_pred < len(traj_x):
            x_slices.append(traj_x[j - seq_O:j + seq_pred])
            u_slices.append(traj_u[j - seq_O:j + seq_pred - 1])
            j += 1

    x_slices = np.array(x_slices, dtype=np.float32)
    u_slices = np.array(u_slices, dtype=np.float32)

    # normalize using saved statistics when available
    if os.path.exists(args['shift_x']) and os.path.exists(args['scale_x']):
        shift_x = np.loadtxt(args['shift_x'])
        scale_x = np.loadtxt(args['scale_x'])
        shift_u = np.loadtxt(args['shift_u'])
        scale_u = np.loadtxt(args['scale_u'])
        scale_x[np.where(scale_x == 0)] = 1
        scale_u[np.where(scale_u == 0)] = 1
        x_slices = (x_slices - shift_x) / scale_x
        u_slices = (u_slices - shift_u) / scale_u

    dataset = torch.utils.data.TensorDataset(torch.tensor(x_slices), torch.tensor(u_slices))
    mid = len(dataset) // 5 if len(dataset) > 5 else max(1, len(dataset) // 2)
    train_sim = HybridDataset(dataset, sim_states=x_slices, has_sim=True)
    val_sim = HybridDataset(torch.utils.data.Subset(dataset, list(range(mid))), sim_states=x_slices[:mid], has_sim=True)
    return train_sim, val_sim

def train(args,model,env,i):
    # print(condition.model)
    if not args['import_saved_data']:
        # model.parameter_restore(args)
        replay_memory = ReplayMemory(args,env, predict_evolution=True)
        #############################00000000000#########################
        x_train = replay_memory.dataset_train
        #############################00000000000#########################
        x_val = replay_memory.val_subset
        x_test = replay_memory.dataset_test
        test_draw = replay_memory.dataset_test_draw

    #
    else:
        x_train   = torch.load(args['SAVE_TRAIN'])
        x_val     = torch.load(args['SAVE_VAL'])
        x_test    = torch.load(args['SAVE_TEST'])
        test_draw = torch.load(args['SAVE_DRAW'])

    if args.get('training_mode','standard') == 'hybrid':
        hybrid_sets = build_hybrid_rollouts(args)
        if hybrid_sets:
            sim_train, sim_val = hybrid_sets
            x_train = ConcatDataset([HybridDataset(x_train, has_sim=False), sim_train])
            x_val = ConcatDataset([HybridDataset(x_val, has_sim=False), sim_val])
        else:
            x_train = HybridDataset(x_train, has_sim=False)
            x_val = HybridDataset(x_val, has_sim=False)
    else:
        x_train = HybridDataset(x_train, has_sim=False)
        x_val = HybridDataset(x_val, has_sim=False)

    ##-------------------是否使用之前参数重新训练------------------##
    args['restore'] = True 
    ##-----------------------------------------------------------##
    if args['restore'] == True:
        model.parameter_restore(args)
        # test_draw = torch.load(args['SAVE_DRAW'])

    test_data = DataLoader(dataset = test_draw, batch_size = 1, shuffle = True, drop_last = False)

    loss = []
    loss_t = []
    for e in range(args['num_epochs']):
        print(f"[epoch {e}] training...", flush=True)
        model.learn(e,x_train,x_val,x_test,args)
        if(e%10==0):
            print("store!!!", flush=True)
            model.parameter_store(args)
        
        if(e%50==0):
            print("draw...", flush=True)
            for x,u in test_data:
                _,_ = model.pred_forward_test(x.float(),u.float(),True,args,e)
        loss.append(model.loss_store)
        loss_t.append(model.loss_store_t)   
        

    fold_path = 'loss/'+condition.method+'/'+str(condition.model)+'/'+str(i)
    if not os.path.exists(fold_path):
        os.makedirs(fold_path)
        
    fold_path_ = fold_path+'/loss_.txt'
    np.savetxt(fold_path_,np.array(loss))
    fold_path_ = fold_path+'/loss_t.txt'
    np.savetxt(fold_path_,np.array(loss_t))  

            
                 


if __name__ == '__main__':
    main()
