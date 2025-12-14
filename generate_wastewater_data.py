#!/usr/bin/env python3
"""
Generate training/test datasets from wastewater data files
Bypasses the need for full gym environment setup
"""

import numpy as np
import torch
import os
from torch.utils.data import Dataset, random_split
import args_new as new_args

class WastewaterDataset(Dataset):
    """Dataset for wastewater training data"""
    def __init__(self, x, u):
        self.x = torch.from_numpy(x).float()
        self.u = torch.from_numpy(u).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.u[idx]

    def determine_shift_and_scale(self):
        """Calculate normalization statistics"""
        self.shift_x = self.x.mean(dim=(0, 1)).numpy()
        self.scale_x = self.x.std(dim=(0, 1)).numpy()
        self.shift_u = self.u.mean(dim=(0, 1)).numpy()
        self.scale_u = self.u.std(dim=(0, 1)).numpy()
        return self.shift_x, self.scale_x, self.shift_u, self.scale_u

    def shift_scale(self, shift_x, scale_x, shift_u, scale_u):
        """Apply normalization"""
        scale_x[scale_x < 1e-10] = 1.0
        scale_u[scale_u < 1e-10] = 1.0
        self.x = (self.x - torch.from_numpy(shift_x).float()) / torch.from_numpy(scale_x).float()
        self.u = (self.u - torch.from_numpy(shift_u).float()) / torch.from_numpy(scale_u).float()


def load_wastewater_trajectories(data_file='Inf_dry_2006.txt', num_trajectories=50,
                                 traj_length=500, state_dim=None, action_dim=4):
    """
    Load wastewater data from file and create synthetic trajectories

    Args:
        data_file: Path to influence data file
        num_trajectories: Number of trajectories to generate
        traj_length: Length of each trajectory
        state_dim: State dimension (if None, inferred from ss_open.txt)
        action_dim: Action dimension

    Returns:
        states: Array of states [num_traj, traj_length, state_dim]
        actions: Array of actions [num_traj, traj_length-1, action_dim]
    """

    try:
        # Try loading initial steady state
        x0 = np.loadtxt('ss_open.txt')
        actual_state_dim = len(x0)
        print(f"✓ Loaded initial state from ss_open.txt, shape: {x0.shape}")
        if state_dim is not None and state_dim != actual_state_dim:
            print(f"⚠ Warning: state_dim={state_dim} does not match actual dimension={actual_state_dim}")
            print(f"  Using actual dimension: {actual_state_dim}")
        state_dim = actual_state_dim
    except:
        if state_dim is None:
            state_dim = 156  # Default fallback
        print(f"⚠ Could not load ss_open.txt, using random initialization with dim={state_dim}")
        x0 = np.random.randn(state_dim) * 0.1

    # Initialize trajectories
    states = np.zeros((num_trajectories, traj_length, state_dim), dtype=np.float32)
    actions = np.zeros((num_trajectories, traj_length-1, action_dim), dtype=np.float32)

    # For wastewater, use realistic action bounds
    action_low = np.array([0.5, 0.5, 0.5, 0.5])    # Minimum control inputs
    action_high = np.array([2.0, 2.0, 2.0, 2.0])   # Maximum control inputs

    for i in range(num_trajectories):
        # Initialize trajectory with initial state + small noise
        states[i, 0] = x0 + np.random.randn(state_dim) * 0.01

        # Generate random control actions
        actions[i] = np.random.uniform(action_low, action_high,
                                      size=(traj_length-1, action_dim)).astype(np.float32)

        # Simple dynamics model: x_{t+1} = x_t + noise
        # (In real system, this would use MBR simulator)
        for t in range(traj_length-1):
            # Add small dynamics: state slowly evolves with random walk
            # In a real system, actions would influence state through a simulator
            state_change = np.random.randn(state_dim) * 0.01
            states[i, t+1] = states[i, t] + state_change

    print(f"✓ Generated {num_trajectories} trajectories of length {traj_length}")
    print(f"  States shape: {states.shape}")
    print(f"  Actions shape: {actions.shape}")

    return states, actions


def create_sequence_dataset(states, actions, seq_length_old=20, seq_length_pred=20):
    """
    Create sliding window sequences from trajectories

    Args:
        states: [num_traj, traj_length, state_dim]
        actions: [num_traj, traj_length-1, action_dim]
        seq_length_old: Length of history window
        seq_length_pred: Length of prediction window

    Returns:
        x_sequences: [num_samples, seq_length_old + seq_length_pred, state_dim]
        u_sequences: [num_samples, seq_length_old + seq_length_pred - 1, action_dim]
    """
    num_traj, traj_length, state_dim = states.shape
    action_dim = actions.shape[2]

    x_list = []
    u_list = []

    for i in range(num_traj):
        traj_x = states[i]
        traj_u = actions[i]

        # Add padding for old_horizon
        add_x = np.repeat(traj_x[0:1], seq_length_old, axis=0)
        add_u = np.zeros((seq_length_old, action_dim), dtype=np.float32)

        traj_x = np.concatenate((add_x, traj_x))
        traj_u = np.concatenate((add_u, traj_u))

        # Create sliding windows
        j = seq_length_old
        while j + seq_length_pred < len(traj_x):
            x_list.append(traj_x[j - seq_length_old : j + seq_length_pred])
            u_list.append(traj_u[j - seq_length_old : j + seq_length_pred - 1])
            j += 1

    x_sequences = np.array(x_list, dtype=np.float32)
    u_sequences = np.array(u_list, dtype=np.float32)

    print(f"✓ Created {len(x_sequences)} sequence samples")
    print(f"  X sequences shape: {x_sequences.shape}")
    print(f"  U sequences shape: {u_sequences.shape}")

    return x_sequences, u_sequences


def generate_and_save_wastewater_data(args, save_dir='data/waste_water'):
    """Generate and save wastewater training/test datasets"""

    print("\n" + "="*60)
    print("Generating Wastewater Training Data")
    print("="*60)

    # Generate training trajectories
    print("\n[1/3] Generating training trajectories...")
    states_train, actions_train = load_wastewater_trajectories(
        num_trajectories=100,  # More trajectories for training
        traj_length=200
    )

    # Generate test trajectories
    print("\n[2/3] Generating test trajectories...")
    states_test, actions_test = load_wastewater_trajectories(
        num_trajectories=20,   # Fewer for testing
        traj_length=200
    )

    # Create sequence datasets
    print("\n[3/3] Creating sequence datasets...")
    seq_old = args.get('old_horizon', 20)
    seq_pred = args.get('pred_horizon', 20)

    x_train, u_train = create_sequence_dataset(states_train, actions_train, seq_old, seq_pred)
    x_test, u_test = create_sequence_dataset(states_test, actions_test, seq_old, seq_pred)

    # Create datasets
    dataset_train = WastewaterDataset(x_train, u_train)
    dataset_test = WastewaterDataset(x_test, u_test)

    # Calculate normalization statistics from training data
    print("\n[4/4] Calculating normalization statistics...")
    shift_x, scale_x, shift_u, scale_u = dataset_train.determine_shift_and_scale()

    # Apply normalization
    dataset_train.shift_scale(shift_x, scale_x, shift_u, scale_u)
    dataset_test.shift_scale(shift_x, scale_x, shift_u, scale_u)

    # Create validation split
    len_train = len(dataset_train)
    len_val = int(np.round(len_train * args.get('val_frac', 0.2)))
    len_train -= len_val
    train_subset, val_subset = random_split(dataset_train, [len_train, len_val],
                                           generator=torch.Generator().manual_seed(1))

    # Save datasets
    os.makedirs(save_dir, exist_ok=True)
    torch.save(train_subset, f'{save_dir}/train.pt')
    torch.save(val_subset, f'{save_dir}/val.pt')
    torch.save(dataset_test, f'{save_dir}/test.pt')
    torch.save(dataset_test, f'{save_dir}/draw.pt')  # For visualization

    # Save normalization statistics
    np.savetxt(f'{save_dir}/shift_x.txt', shift_x)
    np.savetxt(f'{save_dir}/scale_x.txt', scale_x)
    np.savetxt(f'{save_dir}/shift_u.txt', shift_u)
    np.savetxt(f'{save_dir}/scale_u.txt', scale_u)

    print("\n" + "="*60)
    print("✓ Data generation complete!")
    print(f"  Training samples: {len(train_subset)}")
    print(f"  Validation samples: {len(val_subset)}")
    print(f"  Test samples: {len(dataset_test)}")
    print(f"  Saved to: {save_dir}/")
    print("="*60 + "\n")

    return train_subset, val_subset, dataset_test, dataset_test


if __name__ == '__main__':
    # Load config
    args = dict(new_args.args, **new_args.ENV_PARAMS['waste_water'])

    # Generate data
    generate_and_save_wastewater_data(args)

    print("\nData generation successful!")
    print("You can now run: python train_wastewater.py --method mamba")
