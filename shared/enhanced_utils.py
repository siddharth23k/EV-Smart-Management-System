"""
Enhanced Unified EV Smart Management System with validation, error handling, 
batch inference, and quantization support.
"""

import os
import sys
import json
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

# Add project root to path for absolute imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from shared.config import get_config
from modules.braking.models.multitask_lstm_cnn_attention import MultitaskLSTMCNNAttention
from modules.soc.models.lstm_cnn_attention_soc import LSTMCNNAttentionSoC

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InputValidator:
    """Validates input data for the unified pipeline."""
    
    def __init__(self, config):
        self.config = config
        self.braking_config = config.get_braking_model_config()
        self.soc_config = config.get_soc_model_config()
    
    def validate_braking_input(self, driving_window: np.ndarray) -> bool:
        """Validate braking input data."""
        try:
            # Check type
            if not isinstance(driving_window, np.ndarray):
                raise ValueError("Driving window must be numpy array")
            
            # Check shape
            expected_shape = (self.braking_config.get('window_size', 75), 
                           self.braking_config.get('input_dim', 3))
            if driving_window.shape != expected_shape:
                raise ValueError(f"Driving window shape must be {expected_shape}, got {driving_window.shape}")
            
            # Check data type
            if driving_window.dtype != np.float32:
                logger.warning(f"Driving window dtype is {driving_window.dtype}, converting to float32")
                driving_window = driving_window.astype(np.float32)
            
            # Check for NaN or Inf
            if np.any(np.isnan(driving_window)) or np.any(np.isinf(driving_window)):
                raise ValueError("Driving window contains NaN or Inf values")
            
            # Check reasonable ranges
            if np.abs(driving_window).max() > 1000:
                logger.warning("Driving window contains very large values, consider normalization")
            
            return True
            
        except Exception as e:
            logger.error(f"Braking input validation failed: {e}")
            return False
    
    def validate_soc_input(self, battery_window: np.ndarray) -> bool:
        """Validate SoC input data."""
        try:
            # Check type
            if not isinstance(battery_window, np.ndarray):
                raise ValueError("Battery window must be numpy array")
            
            # Check shape
            expected_shape = (self.soc_config.get('window_size', 50), 
                           self.soc_config.get('input_dim', 3))
            if battery_window.shape != expected_shape:
                raise ValueError(f"Battery window shape must be {expected_shape}, got {battery_window.shape}")
            
            # Check data type
            if battery_window.dtype != np.float32:
                logger.warning(f"Battery window dtype is {battery_window.dtype}, converting to float32")
                battery_window = battery_window.astype(np.float32)
            
            # Check for NaN or Inf
            if np.any(np.isnan(battery_window)) or np.any(np.isinf(battery_window)):
                raise ValueError("Battery window contains NaN or Inf values")
            
            # Check reasonable ranges for battery data (normalized)
            # After normalization, values should be roughly in range [-3, 3]
            if battery_window.shape[1] >= 1:  # If voltage channel exists
                voltage = battery_window[:, 0]
                if np.any(np.abs(voltage) > 5):
                    logger.warning("Normalized voltage values seem unusual (abs > 5)")
            
            return True
            
        except Exception as e:
            logger.error(f"SoC input validation failed: {e}")
            return False
    
    def validate_soc_value(self, current_soc: float) -> bool:
        """Validate current SoC value."""
        try:
            if not isinstance(current_soc, (int, float)):
                raise ValueError("Current SoC must be a number")
            
            if not 0 <= current_soc <= 1:
                raise ValueError("Current SoC must be between 0 and 1")
            
            return True
            
        except Exception as e:
            logger.error(f"SoC value validation failed: {e}")
            return False

class ModelQuantizer:
    """Handles model quantization for faster inference."""
    
    @staticmethod
    def quantize_model(model: nn.Module, example_input: torch.Tensor) -> nn.Module:
        """Quantize PyTorch model for faster inference."""
        try:
            # Set model to evaluation mode
            model.eval()
            
            # Create quantized model
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {nn.Linear, nn.LSTM, nn.Conv1d}, 
                dtype=torch.qint8
            )
            
            logger.info("Model quantized successfully")
            return quantized_model
            
        except Exception as e:
            logger.warning(f"Model quantization failed: {e}, using original model")
            return model

class EnhancedEVPipeline:
    """Enhanced EV Smart Management System with all improvements."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize enhanced pipeline."""
        self.config = get_config() if config_path is None else get_config()
        self.device = self.config.get_device()
        
        # Setup paths and configs first
        self.paths = self.config.get_paths_config()
        self.inference_config = self.config.get_inference_config()
        self.performance_config = self.config.get_performance_config()
        
        # Initialize components
        self.validator = InputValidator(self.config)
        self.quantizer = ModelQuantizer()
        
        # Initialize models
        self.braking_model = None
        self.soc_model = None
        self.braking_model_quantized = None
        self.soc_model_quantized = None
        
        # Load models
        self._load_models()
        
        logger.info("Enhanced EV Pipeline initialized successfully")
    
    def _load_models(self):
        """Load models with error handling."""
        try:
            # Load braking model
            self._load_braking_model()
        except Exception as e:
            logger.error(f"Failed to load braking model: {e}")
            raise RuntimeError(f"Braking model loading failed: {e}")
        
        try:
            # Load SoC model
            self._load_soc_model()
        except Exception as e:
            logger.error(f"Failed to load SoC model: {e}")
            raise RuntimeError(f"SoC model loading failed: {e}")
    
    def _load_braking_model(self):
        """Load braking model with error handling."""
        model_path = os.path.join(self.paths['models']['braking'], 'final_multitask_model.pth')
        hp_path = os.path.join(self.paths['models']['braking'], 'best_ga_hyperparams.json')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Braking model not found: {model_path}")
        
        try:
            # Load hyperparameters if available
            if os.path.exists(hp_path):
                with open(hp_path) as f:
                    hp = json.load(f)["hyperparams"]
                # Use parameters that match the saved model architecture
                self.braking_model = MultitaskLSTMCNNAttention(
                    input_dim=3,
                    cnn_channels=32,  # Match saved model
                    lstm_hidden=hp.get("lstm_hidden_size", 64),
                    num_lstm_layers=1,  # Match saved model
                    dropout_rate=0.0,  # Match saved model
                )
            else:
                # Use default configuration
                braking_config = self.config.get_braking_model_config()
                self.braking_model = MultitaskLSTMCNNAttention(
                    input_dim=braking_config.get('input_dim', 3),
                    cnn_channels=braking_config.get('cnn_channels', 32),
                    lstm_hidden=braking_config.get('lstm_hidden', 64),
                    num_lstm_layers=braking_config.get('num_lstm_layers', 1),
                    dropout_rate=braking_config.get('dropout_rate', 0.0),
                )
            
            # Load state dict
            state_dict = torch.load(model_path, map_location=self.device)
            self.braking_model.load_state_dict(state_dict)
            self.braking_model.to(self.device)
            self.braking_model.eval()
            
            # Quantize if enabled
            if self.inference_config.get('quantization', True):
                example_input = torch.randn(1, 75, 3).to(self.device)
                self.braking_model_quantized = self.quantizer.quantize_model(
                    self.braking_model, example_input
                )
            
            logger.info("Braking model loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Error loading braking model: {e}")
    
    def _load_soc_model(self):
        """Load SoC model with error handling."""
        model_path = os.path.join(self.paths['models']['soc'], 'lstm_cnn_attention_soc.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SoC model not found: {model_path}")
        
        try:
            # Use parameters that match the saved model architecture
            # The saved model has different architecture than config defaults
            self.soc_model = LSTMCNNAttentionSoC(
                input_dim=3,
                cnn_channels=64,  # Match saved model
                lstm_hidden=128,  # Match saved model
                num_lstm_layers=2,  # Match saved model
                dropout=0.2,  # Match saved model
            )
            
            # Load state dict
            state_dict = torch.load(model_path, map_location=self.device)
            self.soc_model.load_state_dict(state_dict)
            self.soc_model.to(self.device)
            self.soc_model.eval()
            
            # Quantize if enabled
            if self.inference_config.get('quantization', True):
                example_input = torch.randn(1, 50, 3).to(self.device)
                self.soc_model_quantized = self.quantizer.quantize_model(
                    self.soc_model, example_input
                )
            
            logger.info("SoC model loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Error loading SoC model: {e}")
    
    def run_single(self, driving_window: np.ndarray, battery_window: np.ndarray, 
                  current_soc: float) -> Dict[str, Any]:
        """Run single inference with validation."""
        # Validate inputs
        if self.inference_config.get('validate_inputs', True):
            if not self.validator.validate_braking_input(driving_window):
                raise ValueError("Invalid braking input")
            if not self.validator.validate_soc_input(battery_window):
                raise ValueError("Invalid SoC input")
            if not self.validator.validate_soc_value(current_soc):
                raise ValueError("Invalid current SoC value")
        
        # Convert to tensors
        driving_tensor = torch.tensor(driving_window, dtype=torch.float32).unsqueeze(0).to(self.device)
        battery_tensor = torch.tensor(battery_window, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            # Use quantized models if available
            braking_model = self.braking_model_quantized or self.braking_model
            soc_model = self.soc_model_quantized or self.soc_model
            
            # Braking prediction
            class_logits, intensity = braking_model(driving_tensor)
            class_pred = torch.argmax(class_logits, dim=1).item()
            intensity_val = torch.sigmoid(intensity).item()
            
            # SoC prediction
            soc_pred = soc_model(battery_tensor).item()
            soc_pred = np.clip(soc_pred, 0, 1)  # Ensure valid range
        
        # Process results
        return self._process_results(class_pred, intensity_val, soc_pred, current_soc)
    
    def run_batch(self, driving_windows: List[np.ndarray], battery_windows: List[np.ndarray], 
                 current_socs: List[float]) -> List[Dict[str, Any]]:
        """Run batch inference."""
        if len(driving_windows) != len(battery_windows) or len(driving_windows) != len(current_socs):
            raise ValueError("All input lists must have the same length")
        
        # Validate all inputs
        if self.inference_config.get('validate_inputs', True):
            for i, (dw, bw, soc) in enumerate(zip(driving_windows, battery_windows, current_socs)):
                if not self.validator.validate_braking_input(dw):
                    raise ValueError(f"Invalid braking input at index {i}")
                if not self.validator.validate_soc_input(bw):
                    raise ValueError(f"Invalid SoC input at index {i}")
                if not self.validator.validate_soc_value(soc):
                    raise ValueError(f"Invalid current SoC at index {i}")
        
        # Convert to tensors
        driving_tensor = torch.tensor(np.stack(driving_windows), dtype=torch.float32).to(self.device)
        battery_tensor = torch.tensor(np.stack(battery_windows), dtype=torch.float32).to(self.device)
        
        # Run inference
        with torch.no_grad():
            # Use quantized models if available
            braking_model = self.braking_model_quantized or self.braking_model
            soc_model = self.soc_model_quantized or self.soc_model
            
            # Batch predictions
            class_logits, intensities = braking_model(driving_tensor)
            class_preds = torch.argmax(class_logits, dim=1).cpu().numpy()
            intensity_vals = torch.sigmoid(intensities).cpu().numpy()
            
            soc_preds = soc_model(battery_tensor).cpu().numpy()
            soc_preds = np.clip(soc_preds, 0, 1)  # Ensure valid range
        
        # Process results
        results = []
        for i in range(len(driving_windows)):
            result = self._process_results(
                class_preds[i], intensity_vals[i], soc_preds[i], current_socs[i]
            )
            results.append(result)
        
        return results
    
    def run(self, driving_window: Union[np.ndarray, List[np.ndarray]], 
            battery_window: Union[np.ndarray, List[np.ndarray]], 
            current_soc: Union[float, List[float]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Run inference with automatic batch/single detection."""
        # Determine if batch or single
        if isinstance(driving_window, list):
            return self.run_batch(driving_window, battery_window, current_soc)
        else:
            return self.run_single(driving_window, battery_window, current_soc)
    
    def _process_results(self, class_pred: int, intensity_val: float, 
                       soc_pred: float, current_soc: float) -> Dict[str, Any]:
        """Process inference results into unified output."""
        # Map class to label
        class_labels = ["Light Braking", "Normal Braking", "Emergency Braking"]
        class_label = class_labels[class_pred]
        
        # Calculate energy recovery
        regen_efficiency = self.performance_config.get('regen_efficiency', 0.65)
        energy_recovered = intensity_val * regen_efficiency
        
        # Update SoC
        soc_update_rate = self.performance_config.get('soc_update_rate', 1.0)
        updated_soc = np.clip(current_soc + energy_recovered * soc_update_rate, 0, 1)
        soc_delta = updated_soc - current_soc
        
        # Determine system action
        system_action = self._determine_system_action(class_label, intensity_val, updated_soc)
        
        return {
            "braking": {
                "class": class_label,
                "class_id": class_pred,
                "intensity": intensity_val,
                "confidence": max(0.5, 1.0 - intensity_val)  # Simple confidence estimate
            },
            "energy": {
                "recovered_normalised": energy_recovered,
                "regen_efficiency": regen_efficiency,
                "intensity_raw": intensity_val
            },
            "soc": {
                "estimated": soc_pred,
                "current": current_soc,
                "updated": updated_soc,
                "delta": soc_delta
            },
            "system_action": system_action,
            "metadata": {
                "timestamp": time.time(),
                "device": str(self.device),
                "quantized": self.braking_model_quantized is not None
            }
        }
    
    def _determine_system_action(self, class_label: str, intensity_val: float, 
                               updated_soc: float) -> str:
        """Determine appropriate system action."""
        if class_label == "Emergency Braking":
            return "EMERGENCY: Maximum regenerative braking - safety mode"
        elif class_label == "Normal Braking":
            if updated_soc < 0.2:
                return "REGEN: Normal regenerative braking - battery protection mode"
            else:
                return "REGEN: Normal regenerative braking - efficiency mode"
        else:  # Light Braking
            if updated_soc > 0.9:
                return "REGEN: Light regenerative braking - battery full mode"
            else:
                return "REGEN: Light regenerative braking — comfort mode"
    
    def generate_sample_inputs(self, num_samples: int = 1) -> Tuple[Union[np.ndarray, List[np.ndarray]], 
                                                              Union[np.ndarray, List[np.ndarray]]]:
        """Generate sample inputs for testing."""
        if num_samples == 1:
            return self._generate_single_sample()
        else:
            return self._generate_batch_samples(num_samples)
    
    def _generate_single_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate single sample input."""
        # Driving window: (75, 3) - speed, acceleration, brake pedal
        t = np.linspace(0, 1, 75)
        speed = 60 - 30 * t + np.random.normal(0, 0.5, 75)
        accel = -5 * t + np.random.normal(0, 0.1, 75)
        brake_pedal = 0.3 + 0.5 * t + np.random.normal(0, 0.02, 75)
        brake_pedal = np.clip(brake_pedal, 0, 1)
        driving_window = np.stack([speed, accel, brake_pedal], axis=1).astype(np.float32)
        
        # Normalize
        mean = driving_window.mean(axis=0)
        std = driving_window.std(axis=0) + 1e-8
        driving_window = (driving_window - mean) / std
        
        # Battery window: (50, 3) - voltage, current, temperature
        voltage = 3.8 - 0.5 * np.linspace(0, 1, 50) + np.random.normal(0, 0.01, 50)
        current = -1.0 * np.ones(50) + np.random.normal(0, 0.05, 50)
        temperature = 25 + np.random.normal(0, 0.5, 50)
        battery_window = np.stack([voltage, current, temperature], axis=1).astype(np.float32)
        
        mean = battery_window.mean(axis=0)
        std = battery_window.std(axis=0) + 1e-8
        battery_window = (battery_window - mean) / std
        
        return driving_window, battery_window
    
    def _generate_batch_samples(self, num_samples: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Generate batch sample inputs."""
        driving_windows = []
        battery_windows = []
        
        for _ in range(num_samples):
            dw, bw = self._generate_single_sample()
            driving_windows.append(dw)
            battery_windows.append(bw)
        
        return driving_windows, battery_windows
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and status."""
        return {
            "braking_model_loaded": self.braking_model is not None,
            "soc_model_loaded": self.soc_model is not None,
            "braking_model_quantized": self.braking_model_quantized is not None,
            "soc_model_quantized": self.soc_model_quantized is not None,
            "device": str(self.device),
            "config": {
                "quantization_enabled": self.inference_config.get('quantization', True),
                "input_validation": self.inference_config.get('validate_inputs', True),
                "batch_size": self.inference_config.get('batch_size', 1)
            }
        }
