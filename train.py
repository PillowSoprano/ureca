from replay_fouling import ReplayMemory
# from variant import *

# from three_tanks import three_tank_system as dreamer
# from MBR import MBR as dreamer

from torch.utils.data import Dataset, DataLoader, random_split
import gym

import numpy as np
import torch
import os

import argparse
from data_loader import build_wastewater_datasets


parparser = argparse.ArgumentParser()
parparser.add_argument('method',type=str)
parparser.add_argument('model',type=str)
parparser.add_argument('--dataset', type=str, default='sim', choices=['sim', 'wastewater', 'wastewater-sim'])
parparser.add_argument('--influent-profile', type=str, default='dry')

condition = parparser.parse_args()


def main():
    import args_new as new_args
    args = new_args.args
    dataset_flag = condition.dataset

    if condition.model == 'cartpole':
        from envs.cartpole import CartPoleEnv_adv as dreamer
    if condition.model == 'cartpole_V':
        from envs.cartpole_V import CartPoleEnv_adv as dreamer

    if condition.model == 'half_cheetah':
        dreamer = gym.make('HalfCheetah-v2')
    args['env'] = condition.model



    args = dict(args,**new_args.ENV_PARAMS.get(condition.model, {}))


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

    wastewater_sets = None
    if condition.dataset.startswith('wastewater'):
        args = dict(args, **new_args.ENV_PARAMS['wastewater'])
        args['env'] = 'wastewater'
        if condition.dataset == 'wastewater':
            wastewater_sets = build_wastewater_datasets(
                steady_path=new_args.WASTEWATER_DATA['steady_state_path'],
                influent_paths=new_args.WASTEWATER_DATA['influent_paths'],
                profile_key=condition.influent_profile,
                seq_length=new_args.WASTEWATER_DATA['seq_length'],
                prediction_horizons=new_args.WASTEWATER_DATA['prediction_horizons'],
                train_frac=new_args.WASTEWATER_DATA['train_frac'],
                val_frac=new_args.WASTEWATER_DATA['val_frac'],
                expected_state_dim=new_args.WASTEWATER_DATA['expected_state_dim'],
                expected_influent_dim=new_args.WASTEWATER_DATA['expected_influent_dim'],
                normalize=new_args.WASTEWATER_DATA['normalize'],
            )
        else:
            from waste_water_system import waste_water_system as dreamer

    for i in range(10):
        if condition.dataset == 'wastewater':
            env = None
            args['state_dim'] = new_args.WASTEWATER_DATA['expected_state_dim']
            args['act_dim'] = new_args.WASTEWATER_DATA['expected_influent_dim']
            args['control'] = False
        else:
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
        train(args,model,env,i,wastewater_sets)

        if not args['continue_training']:
            break

def train(args,model,env,i,wastewater_sets=None):
    # print(condition.model)
    if wastewater_sets is not None:
        x_train, x_val, x_test, test_draw = wastewater_sets
    elif not args['import_saved_data']:
        replay_memory = ReplayMemory(args,env, predict_evolution=True)
        x_train = replay_memory.dataset_train
        x_val = replay_memory.val_subset
        x_test = replay_memory.dataset_test
        test_draw = replay_memory.dataset_test_draw
    else:
        x_train   = torch.load(args['SAVE_TRAIN'])
        x_val     = torch.load(args['SAVE_VAL'])
        x_test    = torch.load(args['SAVE_TEST'])
        test_draw = torch.load(args['SAVE_DRAW'])

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
            for batch in test_data:
                x, u = batch[0], batch[1]
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
