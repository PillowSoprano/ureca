#!/usr/bin/env python3
"""
Simplified training script for wastewater treatment system
Trains MamKO and KoVAE models on wastewater data without requiring full gym environment
"""

import os
import sys
import numpy as np
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
import args_new as new_args
from replay_fouling import ReplayMemory
from logger import configure, logkv, dumpkvs

# Command line arguments
parser = argparse.ArgumentParser(description='Train models on wastewater data')
parser.add_argument('--method', type=str, required=True,
                   choices=['mamba', 'kovae'],
                   help='Method to use: mamba (MamKO) or kovae (KoVAE)')
parser.add_argument('--mode', type=str, default='standard',
                   choices=['standard', 'hybrid'],
                   help='Training mode: standard or hybrid')
parser.add_argument('--epochs', type=int, default=None,
                   help='Number of training epochs (default: from config)')
parser.add_argument('--batch_size', type=int, default=None,
                   help='Batch size (default: from config)')
args_cmd = parser.parse_args()

def main():
    # Load configuration
    args = dict(new_args.args, **new_args.ENV_PARAMS['waste_water'])
    args['env'] = 'waste_water'
    args['method'] = args_cmd.method
    args['training_mode'] = args_cmd.mode

    # Override with command line arguments if provided
    if args_cmd.epochs is not None:
        args['num_epochs'] = args_cmd.epochs
    if args_cmd.batch_size is not None:
        args['batch_size'] = args_cmd.batch_size

    # Set state and action dimensions from wastewater config
    args['state_dim'] = 145  # Wastewater system has 145 state variables (from ss_open.txt)
    args['act_dim'] = 4      # 4 control inputs
    args['control'] = False

    # Create save directories
    fold_path = f'save_model/{args_cmd.method}/waste_water'
    if not os.path.exists(fold_path):
        os.makedirs(fold_path)

    args['save_model_path'] = f'{fold_path}/model.pt'
    args['save_opti_path'] = f'{fold_path}/opti.pt'
    args['shift_x'] = f'{fold_path}/shift_x.txt'
    args['scale_x'] = f'{fold_path}/scale_x.txt'
    args['shift_u'] = f'{fold_path}/shift_u.txt'
    args['scale_u'] = f'{fold_path}/scale_u.txt'

    # Import model based on method
    if args_cmd.method == 'mamba':
        from MamKO import Koopman_Desko
        print("=" * 60)
        print("Training MamKO model on wastewater data")
        print("=" * 60)
    elif args_cmd.method == 'kovae':
        from kovae_model import Koopman_Desko
        print("=" * 60)
        print("Training KoVAE model on wastewater data")
        print("=" * 60)

    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Method: {args['method']}")
    print(f"  Training mode: {args['training_mode']}")
    print(f"  State dim: {args['state_dim']}")
    print(f"  Action dim: {args['act_dim']}")
    print(f"  Batch size: {args['batch_size']}")
    print(f"  Learning rate: {args['lr1']}")
    print(f"  Epochs: {args['num_epochs']}")
    print(f"  Latent dim: {args['latent_dim']}")
    print(f"  Prediction horizon: {args['pred_horizon']}")
    print(f"  Control horizon: {args['control_horizon']}")

    # Initialize model
    model = Koopman_Desko(args)

    # Training loop
    print(f"\nStarting training...")
    print(f"Model will be saved to: {args['save_model_path']}")

    # Check if data exists, if not generate it
    data_dir = 'data/waste_water'
    if not os.path.exists(f'{data_dir}/train.pt'):
        print("\nData not found. Generating wastewater training data...")
        from generate_wastewater_data import generate_and_save_wastewater_data
        generate_and_save_wastewater_data(args, data_dir)
        print("Data generation complete!\n")

    # Load pre-generated data
    print("Loading training data...")
    train_data = torch.load(f'{data_dir}/train.pt', weights_only=False)
    val_data = torch.load(f'{data_dir}/val.pt', weights_only=False)
    test_data = torch.load(f'{data_dir}/test.pt', weights_only=False)
    draw_data = torch.load(f'{data_dir}/draw.pt', weights_only=False)

    # Handle both old (Dataset object) and new (dict) formats
    if isinstance(train_data, dict):
        from torch.utils.data import TensorDataset
        x_train = TensorDataset(train_data['x'], train_data['u'])
        x_val = TensorDataset(val_data['x'], val_data['u'])
        x_test = TensorDataset(test_data['x'], test_data['u'])
        test_draw = TensorDataset(draw_data['x'], draw_data['u'])
    else:
        x_train = train_data
        x_val = val_data
        x_test = test_data
        test_draw = draw_data

    # Load normalization statistics
    args['shift_x'] = f'{data_dir}/shift_x.txt'
    args['scale_x'] = f'{data_dir}/scale_x.txt'
    args['shift_u'] = f'{data_dir}/shift_u.txt'
    args['scale_u'] = f'{data_dir}/scale_u.txt'

    print(f"✓ Loaded {len(x_train)} training samples")
    print(f"✓ Loaded {len(x_val)} validation samples")
    print(f"✓ Loaded {len(x_test)} test samples\n")

    try:
        # For hybrid mode, we would need to generate simulator rollouts
        # For now, standard mode training only uses real data
        if args.get('training_mode','standard') == 'hybrid':
            print("⚠  Warning: Hybrid mode not yet implemented for wastewater.")
            print("   Falling back to standard mode training.\n")
            args['training_mode'] = 'standard'

        # Restore model parameters if they exist
        args['restore'] = True
        if args['restore'] and os.path.exists(args['save_model_path']):
            print(f"Restoring model from: {args['save_model_path']}")
            model.parameter_restore(args)

        test_data = DataLoader(dataset=test_draw, batch_size=1, shuffle=True, drop_last=False)

        # Training loop
        loss = []
        loss_t = []

        for e in range(args['num_epochs']):
            print(f"\n[Epoch {e}/{args['num_epochs']}] Training...", flush=True)
            model.learn(e, x_train, x_val, x_test, args)
            maybe_step_scheduler(model, args, metric=model.loss_store_t)

            if e % 10 == 0:
                print("  Saving model checkpoint...", flush=True)
                model.parameter_store(args)

            if e % 50 == 0 and e > 0:
                print("  Generating test predictions...", flush=True)
                for x, u in test_data:
                    _, _ = model.pred_forward_test(x.float(), u.float(), True, args, e)

            loss.append(model.loss_store)
            loss_t.append(model.loss_store_t)

        # Save training history
        loss_dir = f'loss/{args_cmd.method}/waste_water/0'
        os.makedirs(loss_dir, exist_ok=True)
        np.savetxt(f'{loss_dir}/loss_.txt', np.array(loss))
        np.savetxt(f'{loss_dir}/loss_t.txt', np.array(loss_t))

        print("\n" + "=" * 60)
        print("✓ Training completed successfully!")
        print(f"✓ Model saved to: {args['save_model_path']}")
        print(f"✓ Loss history saved to: {loss_dir}/")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
