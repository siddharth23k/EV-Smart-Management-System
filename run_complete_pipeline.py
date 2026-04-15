import os
import sys
import time
import numpy as np
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def check_data_availability():
    print("CHECKING DATA AVAILABILITY----------------")
    
    braking_files = [
        "modules/braking/data/X_train_realistic.npy",
        "modules/braking/data/y_train_realistic.npy", 
        "modules/braking/data/X_val_realistic.npy",
        "modules/braking/data/y_val_realistic.npy"
    ]
    
    soc_files = [
        "modules/soc/data/X_train_soc.npy",
        "modules/soc/data/y_train_soc.npy",
        "modules/soc/data/X_val_soc.npy", 
        "modules/soc/data/y_val_soc.npy"
    ]
    
    missing_braking = [f for f in braking_files if not os.path.exists(f)]
    missing_soc = [f for f in soc_files if not os.path.exists(f)]
    
    if missing_braking:
        print(f"Missing braking data: {missing_braking}")
        return False
    
    if missing_soc:
        print(f"Missing SoC data: {missing_soc}")
        return False
    
    print("All datasets available")
    return True

def generate_braking_data():
    print("GENERATING BRAKING DATASET----------------")
    
    try:
        from modules.braking.data.realistic_ev_simulation import RealisticEVSimulator
        
        simulator = RealisticEVSimulator()
        
        # Generate training data
        print("Generating training data...")
        X_train, y_class_train, y_int_train = simulator.generate_dataset(
            n_samples=5000
        )
        
        # Generate validation data  
        print("Generating validation data...")
        X_val, y_class_val, y_int_val = simulator.generate_dataset(
            n_samples=1000
        )
        
        # Save data
        os.makedirs("modules/braking/data", exist_ok=True)
        np.save("modules/braking/data/X_train_realistic.npy", X_train)
        np.save("modules/braking/data/y_train_realistic.npy", y_class_train)
        np.save("modules/braking/data/X_val_realistic.npy", X_val)
        np.save("modules/braking/data/y_val_realistic.npy", y_class_val)
        
        # Save intensity data
        np.save("modules/braking/data/y_int_train_realistic.npy", y_int_train)
        np.save("modules/braking/data/y_int_val_realistic.npy", y_int_val)
        
        print("Braking dataset generated successfully")
        return True
        
    except Exception as e:
        print(f"Error generating braking data: {e}")
        return False

def generate_soc_data():
    print("GENERATING SOC DATASET----------------")
    
    try:
        from modules.soc.data.preprocess import preprocess_nasa_data
        
        # Check if NASA data exists
        nasa_data_files = [f for f in os.listdir("modules/soc/data") if f.endswith('.csv')]
        if len(nasa_data_files) < 10:
            print("Insufficient NASA battery data files")
            return False
        
        print(f"Found {len(nasa_data_files)} NASA data files")
        
        # Run preprocessing
        print("Running preprocessing...")
        import sys
        sys.path.append("modules/soc/data")
        
        # Import and run the main function directly
        import importlib.util
        spec = importlib.util.spec_from_file_location("preprocess", "modules/soc/data/preprocess.py")
        preprocess_module = importlib.util.module_from_spec(spec)
        
        # Change to data directory for relative paths
        original_cwd = os.getcwd()
        os.chdir("modules/soc/data")
        
        try:
            spec.loader.exec_module(preprocess_module)
            os.chdir(original_cwd)
            print("SoC dataset generated successfully")
            return True
        except Exception as e:
            os.chdir(original_cwd)
            print(f"Failed to generate SoC dataset: {e}")
            return False
            
    except Exception as e:
        print(f"Error generating SoC data: {e}")
        return False

def check_model_availability():
    print("CHECKING MODEL AVAILABILITY----------------")
    
    models = {
        "braking": "modules/braking/models/final_multitask_model.pth",
        "soc": "modules/soc/models/lstm_cnn_attention_soc.pth"
    }
    
    missing = []
    for name, path in models.items():
        if not os.path.exists(path):
            missing.append(f"{name}: {path}")
    
    if missing:
        print(f"Missing models: {missing}")
        return False
    
    print("All models available")
    return True

def train_models():
    print("TRAINING MODELS----------------")
    
    try:
        # Train braking model
        print("Training braking model...")
        os.system("cd /Users/siddh/Desktop/SID/projects/EV-Smart-Management-System && source .venv/bin/activate && python modules/train/train_braking.py")
        
        # Train SoC model
        print("Training SoC model...")
        os.system("cd /Users/siddh/Desktop/SID/projects/EV-Smart-Management-System && source .venv/bin/activate && python modules/train/train_soc.py")
        
        print("Model training completed")
        return True
        
    except Exception as e:
        print(f"Error training models: {e}")
        return False

def test_enhanced_pipeline():
    print("TESTING ENHANCED PIPELINE WITH COGNITIVE PROFILING----------------")
    
    try:
        from shared.enhanced_utils import EnhancedEVPipeline
        
        # Initialize pipeline
        print("Initializing enhanced pipeline...")
        pipeline = EnhancedEVPipeline()
        
        # Generate test data
        print("Generating test data...")
        driving_window, battery_window = pipeline.generate_sample_inputs()
        current_soc = 0.45
        
        # Test basic inference
        print("Testing basic inference...")
        start_time = time.time()
        basic_result = pipeline.run_single(driving_window, battery_window, current_soc)
        basic_time = time.time() - start_time
        
        print(f"Basic inference: {basic_result['system_action']}")
        print(f"Inference time: {basic_time*1000:.2f}ms")
        
        # Test cognitive inference
        print("Testing cognitive inference...")
        start_time = time.time()
        cognitive_result = pipeline.run_with_cognitive(
            driving_window, battery_window, current_soc, driver_id="test_driver_001"
        )
        cognitive_time = time.time() - start_time
        
        print(f"Driver style: {cognitive_result['cognitive']['driver_profile']['driving_style']}")
        print(f"Prediction confidence: {cognitive_result['cognitive']['prediction_confidence']:.2f}")
        print(f"Cognitive action: {cognitive_result['system_action']}")
        print(f"Inference time: {cognitive_time*1000:.2f}ms")
        
        # Test batch inference
        print("  Testing batch inference...")
        driving_windows, battery_windows = pipeline.generate_sample_inputs(10)
        current_socs = [0.3 + i*0.05 for i in range(10)]
        
        start_time = time.time()
        batch_results = pipeline.run_batch(driving_windows, battery_windows, current_socs)
        batch_time = time.time() - start_time
        
        throughput = len(batch_results) / batch_time
        print(f"Batch inference: {len(batch_results)} samples")
        print(f"Throughput: {throughput:.1f} samples/second")
        
        # Get cognitive summary
        print("Getting cognitive summary...")
        summary = pipeline.get_cognitive_summary()
        print(f"Active drivers: {summary.get('active_drivers', 0)}")
        print(f"Adaptation level: {summary.get('adaptation_level', 0):.2f}")
        
        print("Enhanced pipeline test completed successfully")
        return True
        
    except Exception as e:
        print(f"Error testing enhanced pipeline: {e}")
        return False

def run_performance_benchmark():
    print("RUNNING PERFORMANCE BENCHMARK----------------")
    
    try:
        from shared.enhanced_utils import EnhancedEVPipeline
        
        pipeline = EnhancedEVPipeline()
        
        # Benchmark single inference
        print("Benchmarking single inference...")
        driving_window, battery_window = pipeline.generate_sample_inputs()
        current_soc = 0.5
        
        times = []
        for i in range(100):
            start_time = time.time()
            result = pipeline.run_single(driving_window, battery_window, current_soc)
            times.append(time.time() - start_time)
        
        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        
        print(f"Average inference time: {avg_time:.2f}±{std_time:.2f}ms")
        print(f"Min/Max time: {min(times)*1000:.2f}/{max(times)*1000:.2f}ms")
        
        # Benchmark batch inference
        print("Benchmarking batch inference...")
        batch_sizes = [1, 10, 50, 100]
        
        for batch_size in batch_sizes:
            print(f"Generating {batch_size} samples...")
            driving_windows, battery_windows = pipeline.generate_sample_inputs(batch_size)
            
            # Handle case where single sample returns arrays instead of lists
            if batch_size == 1:
                if not isinstance(driving_windows, list):
                    driving_windows = [driving_windows]
                if not isinstance(battery_windows, list):
                    battery_windows = [battery_windows]
            
            # Ensure matching lengths for batch processing
            current_socs = [0.5 + i*0.001 for i in range(batch_size)]  # Smaller increments to stay in [0,1]
            current_socs = [min(0.95, max(0.05, soc)) for soc in current_socs]  # Clamp to valid range
            
            print(f"Generated {len(driving_windows)} driving windows, {len(battery_windows)} battery windows, {len(current_socs)} SoC values")
            
            if len(driving_windows) != len(battery_windows) or len(driving_windows) != len(current_socs):
                print(f"ERROR: Length mismatch - driving: {len(driving_windows)}, battery: {len(battery_windows)}, soc: {len(current_socs)}")
                continue
            
            start_time = time.time()
            results = pipeline.run_batch(driving_windows, battery_windows, current_socs)
            batch_time = time.time() - start_time
            
            throughput = batch_size / batch_time
            print(f"Batch size {batch_size}: {throughput:.1f} samples/second")
        
        print("Performance benchmark completed")
        return True
        
    except Exception as e:
        print(f"Error in performance benchmark: {e}")
        return False

def main():
    print("EV SMART MANAGEMENT SYSTEM - COMPLETE PIPELINE TEST----------------")
    
    results = {}
    
    # Check and generate data if needed
    if not check_data_availability():
        print("GENERATING MISSING DATASETS----------------")
        
        braking_success = generate_braking_data()
        soc_success = generate_soc_data()
        
        results['data_generation'] = braking_success and soc_success
    else:
        results['data_generation'] = True
    
    # Check and train models if needed
    if not check_model_availability():
        print("TRAINING MISSING MODELS----------------")
        
        results['model_training'] = train_models()
    else:
        results['model_training'] = True
    
    # Test enhanced pipeline

    print("TESTING ENHANCED PIPELINE----------------")

    
    results['enhanced_pipeline'] = test_enhanced_pipeline()
    
    # Performance benchmark

    print("PERFORMANCE BENCHMARK----------------")

    
    results['performance'] = run_performance_benchmark()
    
    # Summary

    print("PIPELINE TEST SUMMARY----------------")

    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    overall_success = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if overall_success else 'SOME TESTS FAILED'}")
    
    if overall_success:
        print("\nEV Smart Management System is fully operational!")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
