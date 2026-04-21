#!/usr/bin/env python3
"""
Evaluate existing trained models to calculate and display their quality metrics.
This script loads the existing trained models and evaluates them on validation data
to show the actual performance metrics like RMSE, MAE, accuracy, etc.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from shared.config import get_config
from shared.dataset_loader import get_dataset_loader
from modules.braking.models.multitask_lstm_cnn_attention import MultitaskLSTMCNNAttention
from modules.soc.models.lstm_cnn_attention_soc import LSTMCNNAttentionSoC

def evaluate_braking_model():
    """Evaluate the existing braking model and calculate metrics."""
    print("Evaluating braking model...")
    
    try:
        config = get_config()
        dataset_loader = get_dataset_loader()
        
        # Load validation data
        (X_train, X_val, X_test, 
         y_int_train, y_int_val, y_int_test,
         y_class_train, y_class_val, y_class_test) = dataset_loader.load_braking_dataset()
        
        # Load model
        device = torch.device("cpu")
        model = MultitaskLSTMCNNAttention(
            input_dim=7,
            cnn_channels=32,
            lstm_hidden=64,
            num_lstm_layers=1,
            dropout_rate=0.0
        )
        
        model_path = "modules/braking/models/final_multitask_model.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Evaluate on validation set
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_class_val_tensor = torch.tensor(y_class_val, dtype=torch.long).to(device)
        
        with torch.no_grad():
            cls_outputs, _ = model(X_val_tensor)
            _, predicted = torch.max(cls_outputs.data, 1)
            
            # Convert to numpy for sklearn metrics
            y_true = y_class_val_tensor.cpu().numpy()
            y_pred = predicted.cpu().numpy()
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average='macro')
            f1_weighted = f1_score(y_true, y_pred, average='weighted')
            
            metrics = {
                'val_accuracy': float(accuracy),
                'val_f1_macro': float(f1_macro),
                'val_f1_weighted': float(f1_weighted),
                'model_type': 'multitask_lstm_cnn_attention',
                'validation_samples': len(y_true)
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Score (Macro): {f1_macro:.4f}")
            print(f"  F1-Score (Weighted): {f1_weighted:.4f}")
            
            return metrics
            
    except Exception as e:
        print(f"Error evaluating braking model: {e}")
        return None

def evaluate_soc_model():
    """Evaluate the existing SoC model and calculate metrics."""
    print("Evaluating SoC model...")
    
    try:
        config = get_config()
        dataset_loader = get_dataset_loader()
        
        # Load validation data
        X_train, y_train, X_val, y_val, X_test, y_test = dataset_loader.load_soc_dataset()
        
        # Load model
        device = torch.device("cpu")
        model = LSTMCNNAttentionSoC(
            input_dim=3,
            cnn_channels=64,
            lstm_hidden=128,
            num_lstm_layers=2,
            dropout=0.2
        )
        
        model_path = "modules/soc/models/lstm_cnn_attention_soc.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Evaluate on validation set
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            outputs = model(X_val_tensor)
            y_pred = outputs.cpu().numpy()
            
            # Calculate regression metrics
            mse = mean_squared_error(y_val, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_val, y_pred)
            mape = np.mean(np.abs((y_val - y_pred) / (y_val + 1e-8))) * 100
            
            metrics = {
                'val_rmse': float(rmse),
                'val_mae': float(mae),
                'val_mape': float(mape),
                'val_mse': float(mse),
                'model_type': 'lstm_cnn_attention_soc',
                'validation_samples': len(y_val)
            }
            
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  MAPE: {mape:.2f}%")
            
            return metrics
            
    except Exception as e:
        print(f"Error evaluating SoC model: {e}")
        return None

def save_metrics_for_pipeline(braking_metrics, soc_metrics):
    """Save calculated metrics so the pipeline can display them."""
    
    try:
        # Save braking metrics
        if braking_metrics:
            braking_dir = Path("modules/braking/models")
            braking_dir.mkdir(parents=True, exist_ok=True)
            with open(braking_dir / "final_multitask_model_metrics.json", 'w') as f:
                json.dump(braking_metrics, f, indent=2)
            print("Braking metrics saved for pipeline display")
        
        # Save SoC metrics
        if soc_metrics:
            soc_dir = Path("modules/soc/models")
            soc_dir.mkdir(parents=True, exist_ok=True)
            with open(soc_dir / "lstm_cnn_attention_soc_metrics.json", 'w') as f:
                json.dump(soc_metrics, f, indent=2)
            print("SoC metrics saved for pipeline display")
            
    except Exception as e:
        print(f"Error saving metrics: {e}")

def main():
    """Main function to evaluate models and save metrics."""
    
    print("evaluating existing trained models")
    
    # Evaluate braking model
    braking_metrics = evaluate_braking_model()
    
    print()
    
    # Evaluate SoC model
    soc_metrics = evaluate_soc_model()
    
    print("\nsummary of model performance")
    
    if braking_metrics:
        print("\nBRAKING MODEL PERFORMANCE:")
        print(f"  Validation Accuracy: {braking_metrics['val_accuracy']:.4f}")
        print(f"  Validation F1-Score (Macro): {braking_metrics['val_f1_macro']:.4f}")
        print(f"  Validation F1-Score (Weighted): {braking_metrics['val_f1_weighted']:.4f}")
        print(f"  Validation Samples: {braking_metrics['validation_samples']}")
    
    if soc_metrics:
        print("\nSOC MODEL PERFORMANCE:")
        print(f"  Validation RMSE: {soc_metrics['val_rmse']:.4f}")
        print(f"  Validation MAE: {soc_metrics['val_mae']:.4f}")
        print(f"  Validation MAPE: {soc_metrics['val_mape']:.2f}%")
        print(f"  Validation Samples: {soc_metrics['validation_samples']}")
    
    # Save metrics for pipeline display
    save_metrics_for_pipeline(braking_metrics, soc_metrics)
    
    print("\nMetrics saved! Now run the pipeline test to see them displayed:")
    print("source .venv/bin/activate && python run_complete_pipeline.py")

if __name__ == "__main__":
    main()
