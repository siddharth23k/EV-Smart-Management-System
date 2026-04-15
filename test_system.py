
# Tests data generation, model training, and unified inference.

import os
import sys
import numpy as np
import torch
import time
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def test_data_generation():
    """Test data generation pipeline."""
    print("=" * 60)
    print("TESTING DATA GENERATION PIPELINE")
    print("=" * 60)
    
    try:
        import modules.data.generate_all_datasets_fixed as data_gen
        print("Running data generation...")
        start_time = time.time()
        data_gen.save_braking_datasets()
        data_gen.save_soc_datasets()
        end_time = time.time()
        print(f"Data generation completed in {end_time - start_time:.2f}s")
        
        # Check if key files exist
        required_files = [
            "modules/braking/data/X_train.npy",
            "modules/braking/data/y_train.npy", 
            "modules/braking/data/X_val.npy",
            "modules/braking/data/y_val.npy",
            "modules/soc/data/X_train_soc.npy",
            "modules/soc/data/y_train_soc.npy",
            "modules/soc/data/X_val_soc.npy",
            "modules/soc/data/y_val_soc.npy"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print(f"ERROR: Missing files: {missing_files}")
            return False
        else:
            print("All required data files generated successfully!")
            
        # Check data shapes
        X_train = np.load("modules/braking/data/X_train.npy")
        y_train = np.load("modules/braking/data/y_train.npy")
        X_train_soc = np.load("modules/soc/data/X_train_soc.npy")
        y_train_soc = np.load("modules/soc/data/y_train_soc.npy")
        
        print(f"Braking data shapes: X={X_train.shape}, y={y_train.shape}")
        print(f"SoC data shapes: X={X_train_soc.shape}, y={y_train_soc.shape}")
        
        return True
        
    except Exception as e:
        print(f"ERROR in data generation: {e}")
        return False

def test_model_training():
    """Test model training pipeline."""
    print("\n" + "=" * 60)
    print("TESTING MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    try:
        device = torch.device("cpu")  # Use CPU for testing
        
        # Test braking training
        print("Testing braking model training...")
        start_time = time.time()
        
        import subprocess
        result = subprocess.run([
            sys.executable, "modules/train/train_braking.py", 
            "--baseline", "--device", str(device)
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"Braking training completed in {end_time - start_time:.2f}s")
            
            # Check if model was saved
            if os.path.exists("modules/braking/models/lstm_cnn_attention_baseline.pth"):
                print("Baseline braking model saved successfully!")
            else:
                print("WARNING: Baseline model not found")
        else:
            print(f"ERROR in braking training: {result.stderr}")
            return False
        
        # Test SoC training
        print("Testing SoC model training...")
        start_time = time.time()
        
        result = subprocess.run([
            sys.executable, "modules/train/train_soc.py", 
            "--baseline", "--device", str(device)
        ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
        
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"SoC training completed in {end_time - start_time:.2f}s")
            
            # Check if model was saved
            if os.path.exists("modules/soc/models/lstm_soc_baseline.pth"):
                print("Baseline SoC model saved successfully!")
            else:
                print("WARNING: SoC model not found")
        else:
            print(f"ERROR in SoC training: {result.stderr}")
            return False
        
        return True
        
    except subprocess.TimeoutExpired:
        print("ERROR: Training timed out")
        return False
    except Exception as e:
        print(f"ERROR in model training: {e}")
        return False

def test_unified_inference():
    """Test unified inference pipeline."""
    print("\n" + "=" * 60)
    print("TESTING UNIFIED INFERENCE PIPELINE")
    print("=" * 60)
    
    try:
        from shared.utils import UnifiedEVPipeline
        
        print("Initializing unified pipeline...")
        start_time = time.time()
        pipeline = UnifiedEVPipeline()
        end_time = time.time()
        
        print(f"Pipeline initialization completed in {end_time - start_time:.2f}s")
        
        # Generate test inputs
        print("Generating test inputs...")
        driving_window, battery_window = pipeline.generate_sample_inputs() if hasattr(pipeline, 'generate_sample_inputs') else generate_sample_inputs()
        
        print(f"Input shapes: driving={driving_window.shape}, battery={battery_window.shape}")
        
        # Run inference
        print("Running unified inference...")
        start_time = time.time()
        result = pipeline.run(
            driving_window=driving_window,
            battery_window=battery_window,
            current_soc=0.65
        )
        end_time = time.time()
        
        print(f"Inference completed in {end_time - start_time:.4f}s")
        
        # Validate output structure
        required_keys = ["braking", "energy", "soc", "system_action"]
        for key in required_keys:
            if key not in result:
                print(f"ERROR: Missing output key: {key}")
                return False
        
        # Validate braking output
        braking = result["braking"]
        if not all(k in braking for k in ["class", "intensity"]):
            print("ERROR: Invalid braking output structure")
            return False
        
        # Validate energy output
        energy = result["energy"]
        if not all(k in energy for k in ["recovered_normalised", "regen_efficiency"]):
            print("ERROR: Invalid energy output structure")
            return False
        
        # Validate SoC output
        soc = result["soc"]
        if not all(k in soc for k in ["estimated", "updated", "delta"]):
            print("ERROR: Invalid SoC output structure")
            return False
        
        print("All outputs validated successfully!")
        print(f"Braking class: {braking['class']}")
        print(f"Braking intensity: {braking['intensity']}")
        print(f"Energy recovered: {energy['recovered_normalised']}")
        print(f"SoC estimated: {soc['estimated']}")
        print(f"SoC updated: {soc['updated']}")
        print(f"System action: {result['system_action']}")
        
        return True
        
    except Exception as e:
        print(f"ERROR in unified inference: {e}")
        return False

def generate_sample_inputs():
    """Generate sample inputs for testing."""
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

def test_performance():
    """Test system performance."""
    print("\n" + "=" * 60)
    print("TESTING SYSTEM PERFORMANCE")
    print("=" * 60)
    
    try:
        from shared.utils import UnifiedEVPipeline
        
        pipeline = UnifiedEVPipeline()
        driving_window, battery_window = generate_sample_inputs()
        
        # Test multiple inferences
        num_tests = 10
        times = []
        
        print(f"Running {num_tests} inference tests...")
        
        for i in range(num_tests):
            start_time = time.time()
            result = pipeline.run(
                driving_window=driving_window,
                battery_window=battery_window,
                current_soc=0.65
            )
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"Average inference time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
        print(f"Throughput: {1/avg_time:.2f} inferences/second")
        
        return True
        
    except Exception as e:
        print(f"ERROR in performance testing: {e}")
        return False

def main():
    """Run all tests."""
    print("EV SMART MANAGEMENT SYSTEM - COMPREHENSIVE TESTING")
    print("=" * 60)
    
    tests = [
        ("Data Generation", test_data_generation),
        ("Model Training", test_model_training),
        ("Unified Inference", test_unified_inference),
        ("Performance", test_performance)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"RUNNING: {test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_func()
            results[test_name] = result
            status = "PASS" if result else "FAIL"
            print(f"\n{test_name}: {status}")
        except Exception as e:
            results[test_name] = False
            print(f"\n{test_name}: FAIL - {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! System is ready for production.")
        return True
    else:
        print("Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
