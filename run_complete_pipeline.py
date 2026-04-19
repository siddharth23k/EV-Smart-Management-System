import os
import sys
import time
import numpy as np
import torch
from pathlib import Path

project_root = Path(__file__).parent
sys.path.append(str(project_root))

def check_data_availability():
    print("Checking data availability...")
    
    from shared.config import get_config
    config = get_config()
    paths = config.get_paths_config()
    
    braking_files = [
        os.path.join(paths['data']['braking'], 'X_train_real.npy'),
        os.path.join(paths['data']['braking'], 'y_class_train_real.npy'), 
        os.path.join(paths['data']['braking'], 'X_val_real.npy'),
        os.path.join(paths['data']['braking'], 'y_class_val_real.npy')
    ]
    
    soc_files = [
        os.path.join(paths['data']['soc'], 'X_train_real.npy'),
        os.path.join(paths['data']['soc'], 'y_train_real.npy'),
        os.path.join(paths['data']['soc'], 'X_val_real.npy'), 
        os.path.join(paths['data']['soc'], 'y_val_real.npy')
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
    print("Generating braking dataset...")
    
    try:
        from modules.braking.data.preprocess_real_data import process_all_data
        process_all_data()
        print("Braking dataset generated")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def generate_soc_data():
    print("Generating SoC dataset...")
    
    try:
        from modules.soc.data.preprocess_real_data import process_all_data
        process_all_data()
        print("SoC dataset generated")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def check_model_availability():
    print("Checking model availability...")
    
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
    print("Training models...")
    
    try:
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
    print("Testing enhanced pipeline...")
    
    try:
        from shared.enhanced_utils import EnhancedEVPipeline
        
        print("Initializing pipeline...")
        pipeline = EnhancedEVPipeline()
        
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
        print("Testing batch inference...")
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
        
        print("Enhanced pipeline test completed")
        return True
        
    except Exception as e:
        print(f"Error testing enhanced pipeline: {e}")
        return False

def run_performance_benchmark():
    print("Running performance benchmark...")
    
    try:
        from shared.enhanced_utils import EnhancedEVPipeline
        
        pipeline = EnhancedEVPipeline()
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
        
        print("Benchmarking batch inference...")
        batch_sizes = [1, 10, 50, 100]
        
        for batch_size in batch_sizes:
            print(f"Generating {batch_size} samples...")
            driving_windows, battery_windows = pipeline.generate_sample_inputs(batch_size)
            
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
    print("EV Smart Management System - Pipeline Test")
    
    results = {}
    
    if not check_data_availability():
        print("Generating missing datasets...")
        results['braking_data'] = generate_braking_data()
        results['soc_data'] = generate_soc_data()
    else:
        results['braking_data'] = True
        results['soc_data'] = True
    
    if not check_model_availability():
        print("Training missing models...")
        results['model_training'] = train_models()
    else:
        results['model_training'] = True
    
    if results.get('braking_data', False) or results.get('soc_data', False) or results.get('model_training', False):
        print("Skipping pipeline tests - some steps failed")
        return
    
    print("Testing enhanced pipeline...")
    results['enhanced_pipeline'] = test_enhanced_pipeline()
    
    print("Performance benchmark...")
    results['performance'] = run_performance_benchmark()
    
    print("Pipeline test summary")
    print(f"Data Generation: {'PASS' if results.get('braking_data', False) else 'FAIL'}")
    print(f"Model Training: {'PASS' if results.get('model_training', False) else 'FAIL'}")
    print(f"Enhanced Pipeline: {'PASS' if results.get('enhanced_pipeline', False) else 'FAIL'}")
    print(f"Performance: {'PASS' if results.get('performance', False) else 'FAIL'}")
    
    overall = all([results.get(k, False) for k in ['braking_data', 'model_training', 'enhanced_pipeline', 'performance']])
    print(f"\nOverall: {'ALL TESTS PASSED' if overall else 'SOME TESTS FAILED'}")
    
    if overall:
        print("EV Smart Management System is fully operational!")
    else:
        print("Some tests failed - check logs above")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
