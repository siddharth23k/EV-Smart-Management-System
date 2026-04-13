#!/usr/bin/env python3
"""
run_unified.py  (project root)
Entry point for the Unified EV Smart Management System.
Runs both modules together and shows integrated output.
Uses exact configuration parameters from config system.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path for absolute imports
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from shared.config import get_config
from shared.enhanced_utils import EnhancedEVPipeline

# Check if models exist using config paths
config = get_config()
paths_config = config.get_paths_config()

required_models = [
    os.path.join(paths_config['models']['braking'], "final_multitask_model.pth"),
    os.path.join(paths_config['models']['soc'], "lstm_cnn_attention_soc.pth")
]

if not all(os.path.exists(p) for p in required_models):
    print("⚠️  Missing model weights. Using dummy predictions.")
    print("   Run: python modules/train/train_all_models.py")
    print("   Or run: python run_complete_pipeline.py")



def generate_sample_inputs():
    """Generate synthetic sample inputs using exact config parameters."""
    config = get_config()
    braking_config = config.get_data_config('braking')
    soc_config = config.get_data_config('soc')
    
    # Use exact window sizes from config
    braking_window_size = braking_config.get('window_size', 75)
    soc_window_size = soc_config.get('window_size', 50)
    
    # Driving window: (window_size, 3) — speed, acceleration, brake pedal
    # Simulates emergency braking scenario
    t = np.linspace(0, 1, braking_window_size)
    speed       = 60 - 30 * t + np.random.normal(0, 0.5, braking_window_size)
    accel       = -5 * t + np.random.normal(0, 0.1, braking_window_size)
    brake_pedal = 0.3 + 0.5 * t + np.random.normal(0, 0.02, braking_window_size)
    brake_pedal = np.clip(brake_pedal, 0, 1)
    driving_window = np.stack([speed, accel, brake_pedal], axis=1).astype(np.float32)

    # Normalize (same as training preprocessing)
    mean = driving_window.mean(axis=0)
    std  = driving_window.std(axis=0) + 1e-8
    driving_window = (driving_window - mean) / std

    # Battery window: (window_size, 3) — voltage, current, temperature
    voltage     = 3.8 - 0.5 * np.linspace(0, 1, soc_window_size) + np.random.normal(0, 0.01, soc_window_size)
    current     = -1.0 * np.ones(soc_window_size) + np.random.normal(0, 0.05, soc_window_size)
    temperature = 25 + np.random.normal(0, 0.5, soc_window_size)
    battery_window = np.stack([voltage, current, temperature], axis=1).astype(np.float32)

    mean = battery_window.mean(axis=0)
    std  = battery_window.std(axis=0) + 1e-8
    battery_window = (battery_window - mean) / std

    return driving_window, battery_window


def print_result(result: dict):
    print("\n" + "="*55)
    print("  EV SMART MANAGEMENT SYSTEM — UNIFIED OUTPUT")
    print("="*55)

    b = result["braking"]
    print(f"\n🚗 BRAKING INTENTION MODULE")
    print(f"   Class     : {b['class']}")
    print(f"   Intensity : {b['intensity']:.4f}  (0=light, 1=emergency)")

    e = result["energy"]
    print(f"\n⚡ REGENERATIVE BRAKING")
    print(f"   Energy Recovered : {e['recovered_normalised']:.6f} (normalised)")
    print(f"   Regen Efficiency : {e['regen_efficiency']*100:.0f}%")

    s = result["soc"]
    print(f"\n🔋 BATTERY SOC MODULE")
    print(f"   Estimated SOC : {s['estimated']:.4f}  ({s['estimated']*100:.1f}%)")
    print(f"   Updated SOC   : {s['updated']:.4f}  ({s['updated']*100:.1f}%)")
    print(f"   SOC Delta     : +{s['delta']:.6f}")

    print(f"\n🎯 SYSTEM ACTION")
    print(f"   {result['system_action']}")
    print("="*55 + "\n")


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 UNIFIED EV SMART MANAGEMENT SYSTEM")
    print("=" * 60)
    print("Using exact configuration parameters from config system")
    print("=" * 60)
    
    try:
        print("🔧 Initializing Enhanced EV Pipeline...")
        pipeline = EnhancedEVPipeline()
        
        # Get model info
        info = pipeline.get_model_info()
        print(f"✅ Braking Model: {'Loaded' if info['braking_model_loaded'] else 'Not Found'}")
        print(f"✅ SoC Model: {'Loaded' if info['soc_model_loaded'] else 'Not Found'}")
        print(f"✅ Device: {info['device']}")
        print(f"✅ Quantization: {'Enabled' if info['config']['quantization_enabled'] else 'Disabled'}")
        
        print("\n📊 Generating sample inputs using config parameters...")
        driving_window, battery_window = generate_sample_inputs()
        
        config = get_config()
        braking_config = config.get_data_config('braking')
        soc_config = config.get_data_config('soc')
        
        print(f"Driving window shape : {driving_window.shape} (config: {braking_config.get('window_size', 75)}, {braking_config.get('features', 3)})")
        print(f"Battery window shape : {battery_window.shape} (config: {soc_config.get('window_size', 50)}, {soc_config.get('features', 3)})")
        
        # Run unified inference
        print("\n🔬 Running unified inference...")
        result = pipeline.run(
            driving_window=driving_window,
            battery_window=battery_window,
            current_soc=0.65,   # assume 65% SOC currently
        )
        
        print_result(result)
        
        print("\n✅ Unified system execution completed successfully!")
        
    except Exception as e:
        print(f"❌ System execution failed: {e}")
        print("Please check:")
        print("  1. Configuration file exists and is valid")
        print("  2. Model files are trained and available")
        print("  3. Dependencies are installed")
        sys.exit(1)