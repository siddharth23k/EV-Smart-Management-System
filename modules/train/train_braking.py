#!/usr/bin/env python3
"""
Production training for Braking Intention Models.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from shared.config import get_config
from shared.train_utils import set_seed, create_data_loaders, EarlyStopper, MetricsTracker
from modules.braking.models.lstm_cnn_attention import LSTMCNNAttention
from modules.braking.models.multitask_lstm_cnn_attention import MultitaskLSTMCNNAttention
from modules.braking.models.genetic_algorithm_optimizer import GeneticAlgorithmOptimizer


def train_baseline_model(X_train, y_train, X_val, y_val, device="cpu", config=None):
    print("Training Baseline LSTM-CNN-Attention Model...")
    
    model = LSTMCNNAttention()
    model = model.to(device)
    
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    
    training_config = config.get_training_config() if config else {}
    batch_size = training_config.get('batch_size', 32)
    learning_rate = training_config.get('learning_rate', 0.001)
    epochs = training_config.get('epochs.braking_baseline', 3)
    patience = training_config.get('patience', 2)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    wait = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        val_acc = 100 * correct / total
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            # Save best model
            paths = config.get_paths_config()
            model_path = paths['models']['braking']
            os.makedirs(model_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_path, "lstm_cnn_attention_baseline.pth"))
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    print(" Baseline model training complete!")
    return model


def train_multitask_model(X_train, y_class_train, y_int_train, X_val, y_class_val, y_int_val, device="cpu", config=None):
    """Train multitask model with both classification and regression."""
    print("Training Multitask LSTM-CNN-Attention Model...")
    
    model = MultitaskLSTMCNNAttention()
    model = model.to(device)
    
    # Convert to tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_class_train, dtype=torch.long),
        torch.tensor(y_int_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_class_val, dtype=torch.long),
        torch.tensor(y_int_val, dtype=torch.float32)
    )
    
    # Get training config
    training_config = config.get_training_config() if config else {}
    batch_size = training_config.get('batch_size', 32)
    learning_rate = training_config.get('learning_rate', 0.001)
    epochs = training_config.get('epochs.braking_multitask', 3)
    patience = training_config.get('patience', 2)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    wait = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y_cls, batch_y_int in train_loader:
            batch_x, batch_y_cls, batch_y_int = batch_x.to(device), batch_y_cls.to(device), batch_y_int.to(device)
            
            optimizer.zero_grad()
            cls_outputs, int_outputs = model(batch_x)
            
            loss_cls = criterion_cls(cls_outputs, batch_y_cls)
            loss_reg = criterion_reg(int_outputs, batch_y_int)
            loss = 0.7 * loss_cls + 0.3 * loss_reg  # Weighted loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        cls_correct = 0
        cls_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y_cls, batch_y_int in val_loader:
                batch_x, batch_y_cls, batch_y_int = batch_x.to(device), batch_y_cls.to(device), batch_y_int.to(device)
                cls_outputs, int_outputs = model(batch_x)
                
                loss_cls = criterion_cls(cls_outputs, batch_y_cls)
                loss_reg = criterion_reg(int_outputs, batch_y_int)
                loss = 0.7 * loss_cls + 0.3 * loss_reg
                val_loss += loss.item()
                
                _, predicted = torch.max(cls_outputs.data, 1)
                cls_total += batch_y_cls.size(0)
                cls_correct += (predicted == batch_y_cls).sum().item()
        
        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        val_acc = 100 * cls_correct / cls_total
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            # Save best model
            paths = config.get_paths_config()
            model_path = paths['models']['braking']
            os.makedirs(model_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_path, "final_multitask_model.pth"))
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    print("✅ Multitask model training complete!")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Braking Intention Models")
    parser.add_argument("--baseline", action="store_true", help="Train baseline LSTM-CNN-Attention model only")
    parser.add_argument("--multitask", action="store_true", help="Train multitask model only")
    parser.add_argument("--ga", action="store_true", help="Run genetic algorithm optimization")
    parser.add_argument("--device", default="auto", help="Device to use (auto/cpu/cuda)")
    
    args = parser.parse_args()
    
    # Initialize config
    config = get_config()
    
    # Set device
    if args.device == "auto":
        device = config.get_device()
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(config.get('system.seed', 42))
    
    # Load datasets
    try:
        print("Loading datasets...")
        paths = config.get_paths_config()
        data_path = paths['data']['braking']
        
        # Prioritize realistic EV simulation dataset
        if os.path.exists(os.path.join(data_path, "X_train_realistic.npy")):
            print("Using realistic EV simulation dataset...")
            X_train = np.load(os.path.join(data_path, "X_train_realistic.npy"))
            y_class_train = np.load(os.path.join(data_path, "y_class_train_realistic.npy"))
            y_int_train = np.load(os.path.join(data_path, "y_int_train_realistic.npy"))
            X_val = np.load(os.path.join(data_path, "X_val_realistic.npy"))
            y_class_val = np.load(os.path.join(data_path, "y_class_val_realistic.npy"))
            y_int_val = np.load(os.path.join(data_path, "y_int_val_realistic.npy"))
            
            # For baseline training, use class labels
            y_train = y_class_train
            y_val = y_class_val
            print("✅ Realistic EV simulation dataset loaded")
        else:
            # Fallback to original hard multitask dataset
            print("Using hard multitask dataset...")
            X_train = np.load(os.path.join(data_path, "X_train_hard_mtl.npy"))
            y_class_train = np.load(os.path.join(data_path, "y_class_train_hard_mtl.npy"))
            y_int_train = np.load(os.path.join(data_path, "y_int_train_hard_mtl.npy"))
            X_val = np.load(os.path.join(data_path, "X_val_hard_mtl.npy"))
            y_class_val = np.load(os.path.join(data_path, "y_class_val_hard_mtl.npy"))
            y_int_val = np.load(os.path.join(data_path, "y_int_val_hard_mtl.npy"))
            
            # For baseline training, use class labels
            y_train = y_class_train
            y_val = y_class_val
            print("✅ Hard multitask dataset loaded")
        
        # Ensure intensity labels are available for multitask training
        if y_int_train is None or y_int_val is None:
            print("❌ Intensity labels not found for multitask training")
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        
    except FileNotFoundError as e:
        print(f"❌ Dataset files not found: {e}")
        print("Please run dataset generation first!")
        return
    
    # Train models based on arguments
    if args.baseline or (not args.multitask and not args.ga):
        train_baseline_model(X_train, y_train, X_val, y_val, device, config)
    
    if args.multitask and y_int_train is not None:
        train_multitask_model(X_train, y_train, y_int_train, X_val, y_val, y_int_val, device, config)
    elif args.multitask:
        print("❌ Multitask training requires intensity labels (y_int_*.npy files)")
    
    if args.ga:
        print("Running Genetic Algorithm Optimization...")
        # Change to braking models directory for GA
        os.chdir("modules/braking/models")
        run_ga_optimization()
        os.chdir("../../../")
    
    print("🎉 All training completed!")


if __name__ == "__main__":
    main()

