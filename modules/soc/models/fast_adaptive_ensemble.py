
# Fast Adaptive Ensemble Learning for SoC Estimation


import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules.soc.models.lstm_cnn_attention_soc import LSTMCNNAttentionSoC


class TransformerSoCModel(nn.Module):
    """Lightweight Transformer-based SoC estimation model."""
    
    def __init__(self, input_dim=3, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        x = self.input_projection(x) * np.sqrt(self.d_model)
        x = self.transformer(x)
        x = torch.mean(x, dim=1)
        output = self.regressor(x)
        return output.squeeze(-1)


class PhysicsInformedSoCModel(nn.Module):
    """Lightweight Physics-informed neural network for SoC estimation."""
    
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        
        # Feature extraction layers
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*layers)
        
        # Physics features and final regressor
        self.physics_regressor = nn.Sequential(
            nn.Linear(hidden_dim + 3, 32),  # +3 for physics features
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def _compute_physics_features(self, x):
        """Compute simple physics-based features."""
        voltage = x[:, :, 0]
        current = x[:, :, 1]
        temp = x[:, :, 2]
        
        avg_voltage = torch.mean(voltage, dim=1)
        avg_current = torch.mean(current, dim=1)
        avg_temp = torch.mean(temp, dim=1)
        
        return torch.stack([avg_voltage, avg_current, avg_temp], dim=1)
    
    def forward(self, x):
        # Process sequence
        batch_size, seq_len, _ = x.shape
        aggregated_features = []
        
        for t in range(seq_len):
            timestep_features = self.feature_extractor(x[:, t, :])
            aggregated_features.append(timestep_features)
        
        aggregated_features = torch.mean(torch.stack(aggregated_features, dim=1), dim=1)
        
        # Add physics features
        physics_features = self._compute_physics_features(x)
        combined_features = torch.cat([aggregated_features, physics_features], dim=1)
        
        return self.physics_regressor(combined_features).squeeze(-1)


@dataclass
class EnsembleWeights:
    """Weights for ensemble models."""
    lstm_cnn_weight: float
    transformer_weight: float
    physics_weight: float
    
    def as_array(self) -> np.ndarray:
        return np.array([self.lstm_cnn_weight, self.transformer_weight, self.physics_weight])
    
    def normalize(self):
        """Normalize weights to sum to 1."""
        total = self.lstm_cnn_weight + self.transformer_weight + self.physics_weight
        if total > 0:
            self.lstm_cnn_weight /= total
            self.transformer_weight /= total
            self.physics_weight /= total


class FastAdaptiveEnsembleSoC:
    """Fast adaptive ensemble that uses pre-trained models."""
    
    def __init__(self, device=None):
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Initialize models
        self.lstm_cnn_model = None
        self.transformer_model = None
        self.physics_model = None
        
        # Ensemble weights
        self.weights = EnsembleWeights(0.4, 0.3, 0.3)
    
    def load_pretrained_models(self):
        """Load pre-trained models if available."""
        models_loaded = False
        
        # Try to load LSTM-CNN model
        lstm_path = "modules/soc/models/lstm_cnn_attention_soc.pth"
        if os.path.exists(lstm_path):
            self.lstm_cnn_model = LSTMCNNAttentionSoC(
                input_dim=3, cnn_channels=64, lstm_hidden=128, 
                num_lstm_layers=2, dropout=0.2
            )
            self.lstm_cnn_model.load_state_dict(torch.load(lstm_path, map_location=self.device))
            self.lstm_cnn_model.to(self.device)
            self.lstm_cnn_model.eval()
            print("  Loaded pre-trained LSTM-CNN model")
            models_loaded = True
        
        # Try to load ensemble models
        ensemble_lstm_path = "modules/soc/models/ensemble_lstm_cnn.pth"
        if os.path.exists(ensemble_lstm_path):
            self.lstm_cnn_model = LSTMCNNAttentionSoC(
                input_dim=3, cnn_channels=64, lstm_hidden=128, 
                num_lstm_layers=2, dropout=0.2
            )
            self.lstm_cnn_model.load_state_dict(torch.load(ensemble_lstm_path, map_location=self.device))
            self.lstm_cnn_model.to(self.device)
            self.lstm_cnn_model.eval()
            print("  Loaded ensemble LSTM-CNN model")
            models_loaded = True
        
        # Create lightweight models for ensemble
        if models_loaded:
            self.transformer_model = TransformerSoCModel(input_dim=3, d_model=64, nhead=4, num_layers=2)
            self.transformer_model.to(self.device)
            self.transformer_model.eval()
            
            self.physics_model = PhysicsInformedSoCModel(input_dim=3, hidden_dim=64, num_layers=2)
            self.physics_model.to(self.device)
            self.physics_model.eval()
            
            print("  Created lightweight Transformer and Physics models")
            return True
        
        return False
    
    def quick_train_lightweight_models(self, X_train, y_train, X_val, y_val):
        """Quick training for lightweight models (few epochs)."""
        print("Quick training lightweight models...")
        
        # Simple training function
        def quick_train(model, epochs=3):
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            for epoch in range(epochs):
                # Simple batch training
                batch_size = 128
                for i in range(0, len(X_train), batch_size):
                    batch_x = torch.tensor(X_train[i:i+batch_size], dtype=torch.float32).to(self.device)
                    batch_y = torch.tensor(y_train[i:i+batch_size], dtype=torch.float32).to(self.device)
                    
                    optimizer.zero_grad()
                    pred = model(batch_x)
                    loss = criterion(pred, batch_y)
                    loss.backward()
                    optimizer.step()
            
            model.eval()
        
        # Quick train both models
        quick_train(self.transformer_model, epochs=2)
        quick_train(self.physics_model, epochs=2)
        
        print("  Quick training completed")
    
    def optimize_weights_simple(self, X_val, y_val):
        """Simple weight optimization using validation performance."""
        print("Optimizing ensemble weights...")
        
        # Evaluate each model on validation set
        def evaluate_model(model):
            model.eval()
            predictions = []
            
            with torch.no_grad():
                batch_size = 256
                for i in range(0, len(X_val), batch_size):
                    batch_x = torch.tensor(X_val[i:i+batch_size], dtype=torch.float32).to(self.device)
                    pred = model(batch_x).cpu().numpy()
                    predictions.append(pred)
            
            return np.concatenate(predictions)
        
        # Get predictions from all models
        lstm_pred = evaluate_model(self.lstm_cnn_model)
        transformer_pred = evaluate_model(self.transformer_model)
        physics_pred = evaluate_model(self.physics_model)
        
        # Calculate RMSE for each model
        def rmse(y_true, y_pred):
            return np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        lstm_rmse = rmse(y_val, lstm_pred)
        transformer_rmse = rmse(y_val, transformer_pred)
        physics_rmse = rmse(y_val, physics_pred)
        
        print(f"  Model RMSEs - LSTM-CNN: {lstm_rmse:.4f}, Transformer: {transformer_rmse:.4f}, Physics: {physics_rmse:.4f}")
        
        # Set weights based on inverse RMSE
        inv_rmse_sum = (1/lstm_rmse + 1/transformer_rmse + 1/physics_rmse)
        self.weights.lstm_cnn_weight = (1/lstm_rmse) / inv_rmse_sum
        self.weights.transformer_weight = (1/transformer_rmse) / inv_rmse_sum
        self.weights.physics_weight = (1/physics_rmse) / inv_rmse_sum
        
        print(f"  Optimized weights: LSTM-CNN={self.weights.lstm_cnn_weight:.3f}, "
              f"Transformer={self.weights.transformer_weight:.3f}, "
              f"Physics={self.weights.physics_weight:.3f}")
    
    def forward(self, x):
        """Forward pass through ensemble."""
        if not all([self.lstm_cnn_model, self.transformer_model, self.physics_model]):
            raise RuntimeError("Models not loaded. Call load_pretrained_models() first.")
        
        self.lstm_cnn_model.eval()
        self.transformer_model.eval()
        self.physics_model.eval()
        
        with torch.no_grad():
            lstm_pred = self.lstm_cnn_model(x)
            transformer_pred = self.transformer_model(x)
            physics_pred = self.physics_model(x)
            
            weights_tensor = torch.tensor(self.weights.as_array(), dtype=torch.float32, device=self.device)
            predictions = torch.stack([lstm_pred, transformer_pred, physics_pred], dim=1)
            ensemble_pred = torch.sum(predictions * weights_tensor.unsqueeze(0), dim=1)
            
            return ensemble_pred
    
    def evaluate_ensemble(self, X_test, y_test):
        """Evaluate ensemble performance."""
        with torch.no_grad():
            batch_size = 256
            predictions = []
            
            for i in range(0, len(X_test), batch_size):
                batch_x = torch.tensor(X_test[i:i+batch_size], dtype=torch.float32).to(self.device)
                pred = self.forward(batch_x).cpu().numpy()
                predictions.append(pred)
            
            ensemble_pred = np.concatenate(predictions)
            
            # Calculate RMSE
            ensemble_rmse = np.sqrt(np.mean((y_test - ensemble_pred) ** 2))
            
            return {
                'ensemble_rmse': ensemble_rmse,
                'weights': self.weights.as_array().tolist()
            }
    
    def save_ensemble(self, path="modules/soc/models/fast_adaptive_ensemble.json"):
        """Save ensemble configuration."""
        def convert_types(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, np.float32):
                return float(obj)
            elif isinstance(obj, np.int32):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        ensemble_config = {
            'weights': convert_types(self.weights.as_array()),
            'models_loaded': all([self.lstm_cnn_model, self.transformer_model, self.physics_model])
        }
        
        with open(path, 'w') as f:
            json.dump(ensemble_config, f, indent=4)
        print(f"Fast ensemble configuration saved: {path}")


def create_fast_ensemble():
    """Create fast adaptive ensemble with minimal training."""
    print("=== Creating Fast Adaptive Ensemble SoC Model ===")
    
    # Load data
    DATA = "modules/soc/data"
    X_train = np.load(f"{DATA}/X_train_soc.npy")
    y_train = np.load(f"{DATA}/y_train_soc.npy")
    X_val = np.load(f"{DATA}/X_val_soc.npy")
    y_val = np.load(f"{DATA}/y_val_soc.npy")
    
    # Use smaller dataset for faster processing
    sample_size = 10000
    idx = np.random.choice(len(X_train), sample_size, replace=False)
    X_train = X_train[idx]
    y_train = y_train[idx]
    X_train = X_train[:, :25, :]
    X_val = X_val[:, :25, :]
    
    print(f"Using sample data: {X_train.shape}")
    
    # Create ensemble
    ensemble = FastAdaptiveEnsembleSoC()
    
    # Load pre-trained models
    if ensemble.load_pretrained_models():
        # Quick train lightweight models
        ensemble.quick_train_lightweight_models(X_train, y_train, X_val, y_val)
        
        # Optimize weights
        ensemble.optimize_weights_simple(X_val, y_val)
        
        # Evaluate
        results = ensemble.evaluate_ensemble(X_val, y_val)
        
        print(f"\n=== FAST ENSEMBLE RESULTS ===")
        print(f"Ensemble RMSE: {results['ensemble_rmse']:.4f}")
        print(f"Weights: {results['weights']}")
        
        # Save ensemble
        ensemble.save_ensemble()
        
        return ensemble, results
    else:
        print("No pre-trained models found. Please train models first.")
        return None, None


if __name__ == "__main__":
    ensemble, results = create_fast_ensemble()
    if ensemble:
        print("Fast adaptive ensemble created successfully!")
