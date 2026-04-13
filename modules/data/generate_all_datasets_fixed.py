#!/usr/bin/env python3
"""
Generate ALL datasets needed for EV Smart Management System.
"""

import os
import sys
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from modules.braking.data.generate_dataset import generate_dataset, split_dataset
from modules.braking.data.generate_hard_braking_data import generate_dataset as generate_hard
from modules.braking.data.generate_hard_braking_data_mtl import generate_dataset_mtl as generate_hard_mtl
from modules.braking.data.realistic_ev_simulation import generate_realistic_dataset
from modules.soc.data.preprocess import main as preprocess_soc

def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def save_braking_datasets():
    print("=== Generating Braking Datasets ===")
    
    print("1. Baseline dataset...")
    X, y = generate_dataset()
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)
    np.save('modules/braking/data/X_train.npy', X_train)
    np.save('modules/braking/data/y_train.npy', y_train)
    np.save('modules/braking/data/X_val.npy', X_val)
    np.save('modules/braking/data/y_val.npy', y_val)
    np.save('modules/braking/data/X_test.npy', X_test)
    np.save('modules/braking/data/y_test.npy', y_test)
    
    print("2. Hard braking dataset...")
    generate_hard()
    print("   Hard braking dataset saved")
    
    print("3. Hard multitask dataset (GA-optimized)...")
    X_mtl, y_class_mtl, y_int_mtl = generate_hard_mtl()
    
    idx = np.random.permutation(len(X_mtl))
    X_mtl = X_mtl[idx]
    y_class_mtl = y_class_mtl[idx]
    y_int_mtl = y_int_mtl[idx]
    
    # Split 70/15/15
    n = len(X_mtl)
    n_train = int(0.7 * n)
    n_val = int(0.85 * n)
    
    X_train_mtl, X_val_mtl, X_test_mtl = X_mtl[:n_train], X_mtl[n_train:n_val], X_mtl[n_val:]
    y_class_train_mtl, y_class_val_mtl, y_class_test_mtl = y_class_mtl[:n_train], y_class_mtl[n_train:n_val], y_class_mtl[n_val:]
    y_int_train_mtl, y_int_val_mtl, y_int_test_mtl = y_int_mtl[:n_train], y_int_mtl[n_train:n_val], y_int_mtl[n_val:]
    
    np.save('modules/braking/data/X_train_hard_mtl.npy', X_train_mtl)
    np.save('modules/braking/data/y_class_train_hard_mtl.npy', y_class_train_mtl)
    np.save('modules/braking/data/y_int_train_hard_mtl.npy', y_int_train_mtl)
    np.save('modules/braking/data/X_val_hard_mtl.npy', X_val_mtl)
    np.save('modules/braking/data/y_class_val_hard_mtl.npy', y_class_val_mtl)
    np.save('modules/braking/data/y_int_val_hard_mtl.npy', y_int_val_mtl)
    np.save('modules/braking/data/X_test_hard_mtl.npy', X_test_mtl)
    np.save('modules/braking/data/y_class_test_hard_mtl.npy', y_class_test_mtl)
    np.save('modules/braking/data/y_int_test_hard_mtl.npy', y_int_test_mtl)
    
    # 4. Realistic EV simulation dataset (NEW - primary production dataset)
    print("4. Realistic EV simulation dataset (physics-based)...")
    X_real, y_class_real, y_int_real = generate_realistic_dataset(
        n_samples=15000,
        save_path="modules/braking/data"
    )
    
    print("Braking datasets saved!")

def save_soc_datasets():
    print("=== Generating SoC Datasets ===")
    try:
        preprocess_soc()
        print("SoC datasets saved!")
    except Exception as e:
        print(f"SoC preprocessing failed (NASA CSVs missing?): {e}")
        print("   Synthetic SoC data generation can be added later.")

if __name__ == '__main__':
    save_braking_datasets()
    save_soc_datasets()
    print("\nALL DATASETS GENERATED! Ready for training.")
