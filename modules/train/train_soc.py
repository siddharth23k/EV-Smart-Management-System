import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path for absolute imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from shared.config import get_config
from shared.train_utils import set_seed, create_data_loaders, EarlyStopper, MetricsTracker
from modules.soc.models.lstm_soc import LSTMSoC as LSTMSOC
from modules.soc.models.lstm_cnn_attention_soc import LSTMCNNAttentionSoC, train_soc_model, evaluate_soc_model


def train_lstm_baseline(X_train, y_train, X_val, y_val, device, config=None):
    if device:
        device = device
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Mac GPU
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu") 
    """Train baseline LSTM model for SOC prediction."""
    print("Training Baseline LSTM SOC Model...")
    
    # Check if LSTM SOC model exists, if not create a simple one
    try:
        model = LSTMSOC()
    except:
        # Create a simple LSTM model if LSTMSOC doesn't exist
        class SimpleLSTMSOC(nn.Module):
            def __init__(self, input_dim=3, hidden_dim=64, num_layers=2):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
                self.fc = nn.Sequential(
                    nn.Linear(hidden_dim, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                # Use last time step
                out = self.fc(lstm_out[:, -1, :])
                return out.squeeze(-1)
        
        model = SimpleLSTMSOC()
    
    model = model.to(device)
    
    # Convert to tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    
    # Get training config
    training_config = config.get_training_config() if config else {}
    batch_size = training_config.get('batch_size', 32)
    learning_rate = training_config.get('learning_rate', 0.001)
    epochs = training_config.get('epochs.soc_baseline', 2)
    patience = training_config.get('patience', 2)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.MSELoss()
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
        
        # Validation
        model.eval()
        val_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                predictions.extend(outputs.cpu().numpy())
                targets.extend(batch_y.cpu().numpy())
        
        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        
        val_rmse = np.sqrt(mean_squared_error(targets, predictions))
        val_mae = mean_absolute_error(targets, predictions)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val RMSE: {val_rmse:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            # Save best model
            paths = config.get_paths_config()
            model_path = paths['models']['soc']
            os.makedirs(model_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_path, "lstm_soc_baseline.pth"))
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    print("Baseline LSTM SOC model training complete!")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train SOC Prediction Models")
    parser.add_argument("--baseline", action="store_true", help="Train baseline LSTM model only")
    parser.add_argument("--cnn", action="store_true", help="Train LSTM+CNN+Attention model only")
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
        print("Loading SOC datasets...")
        paths = config.get_paths_config()
        data_path = paths['data']['soc']
        data_config = config.get_data_config('soc')
        subset_size = data_config.get('subset_size', 5000)
        
        X_train = np.load(os.path.join(data_path, "X_train_soc.npy"))[:subset_size]
        y_train = np.load(os.path.join(data_path, "y_train_soc.npy"))[:subset_size]
        X_val = np.load(os.path.join(data_path, "X_val_soc.npy"))[:subset_size//5]
        y_val = np.load(os.path.join(data_path, "y_val_soc.npy"))[:subset_size//5]
        X_test = np.load(os.path.join(data_path, "X_test_soc.npy"))[:subset_size//5]
        y_test = np.load(os.path.join(data_path, "y_test_soc.npy"))[:subset_size//5]
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
        
    except FileNotFoundError as e:
        print(f"Dataset files not found: {e}")
        print("Please run dataset generation first!")
        return
    
    # Train models based on arguments
    if args.baseline or (not args.cnn):
        baseline_model = train_lstm_baseline(X_train, y_train, X_val, y_val, device, config)
        
        # Evaluate baseline model
        baseline_model.eval()
        with torch.no_grad():
            test_predictions = baseline_model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
        
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        test_mae = mean_absolute_error(y_test, test_predictions)
        print(f"\nBaseline LSTM Test Results:")
        print(f"RMSE: {test_rmse:.4f}")
        print(f"MAE: {test_mae:.4f}")
    
    if args.cnn or (not args.baseline):
        print("\nTraining LSTM+CNN+Attention SOC Model...")
        cnn_model = LSTMCNNAttentionSoC()
        
        # Train CNN model using training loop similar to baseline
        train_loader = DataLoader(TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        ), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        ), batch_size=32, shuffle=False)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
        
        best_val_loss = float('inf')
        patience = 2
        wait = 0
        
        for epoch in range(2):  # Fast training
            cnn_model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = cnn_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            cnn_model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = cnn_model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            train_loss /= len(train_loader)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
                # Save best model
                paths = config.get_paths_config()
                model_path = paths['models']['soc']
                os.makedirs(model_path, exist_ok=True)
                torch.save(cnn_model.state_dict(), os.path.join(model_path, "lstm_cnn_attention_soc.pth"))
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Evaluate CNN model
        cnn_results = evaluate_soc_model(cnn_model, X_test, y_test, device=device)
    
    print("All SOC training completed!")


if __name__ == "__main__":
    main()
