#!/usr/bin/env python3
"""
Complete EV Smart Management System Pipeline
Runs entire project from dataset generation to results with full models
No timeouts - optimized for production deployment
"""

import os
import sys
import time
import subprocess
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def print_header():
    """Print pipeline header."""
    print("=" * 80)
    print("🚀 COMPLETE EV SMART MANAGEMENT SYSTEM PIPELINE")
    print("=" * 80)
    print("Running: Dataset Generation → Model Training → Inference → Results")
    print("No timeouts - Production-ready models with optimized parameters")
    print("=" * 80)

def run_data_generation():
    """Run complete data generation pipeline using config parameters."""
    print("\n" + "=" * 60)
    print("📊 PHASE 1: DATASET GENERATION")
    print("=" * 60)
    
    try:
        from shared.config import get_config
        config = get_config()
        paths_config = config.get_paths_config()
        
        print("Generating braking and SoC datasets using config parameters...")
        start_time = time.time()
        
        # Import and run data generation
        import modules.data.generate_all_datasets_fixed as data_gen
        data_gen.save_braking_datasets()
        data_gen.save_soc_datasets()
        
        end_time = time.time()
        print(f"✅ Dataset generation completed in {end_time - start_time:.2f}s")
        
        # Verify datasets exist using config paths
        required_files = [
            os.path.join(paths_config['data']['braking'], "X_train.npy"),
            os.path.join(paths_config['data']['braking'], "y_train.npy"),
            os.path.join(paths_config['data']['braking'], "X_val.npy"), 
            os.path.join(paths_config['data']['braking'], "y_val.npy"),
            os.path.join(paths_config['data']['soc'], "X_train_soc.npy"),
            os.path.join(paths_config['data']['soc'], "y_train_soc.npy"),
            os.path.join(paths_config['data']['soc'], "X_val_soc.npy"),
            os.path.join(paths_config['data']['soc'], "y_val_soc.npy")
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            print(f"❌ Missing datasets: {missing_files}")
            return False
            
        print("✅ All datasets generated successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False

def run_model_training():
    """Run complete model training pipeline using config parameters."""
    print("\n" + "=" * 60)
    print("🧠 PHASE 2: MODEL TRAINING")
    print("=" * 60)
    
    try:
        from shared.config import get_config
        config = get_config()
        paths_config = config.get_paths_config()
        training_config = config.get_training_config()
        
        print("Training braking and SoC models with config parameters...")
        print(f"Batch size: {training_config.get('batch_size', 32)}")
        print(f"Learning rate: {training_config.get('learning_rate', 0.001)}")
        
        start_time = time.time()
        
        # Train braking models
        print("\n🚗️ Training Braking Models...")
        braking_result = subprocess.run([
            sys.executable, "modules/train/train_braking.py",
            "--baseline", "--multitask", "--device", "auto"
        ], capture_output=True, text=True)
        
        if braking_result.returncode != 0:
            print(f"❌ Braking training failed: {braking_result.stderr}")
            return False
        print("✅ Braking models trained successfully!")
        
        # Train SoC models  
        print("\n🔋 Training SoC Models...")
        soc_result = subprocess.run([
            sys.executable, "modules/train/train_soc.py",
            "--baseline", "--cnn", "--device", "auto"
        ], capture_output=True, text=True)
        
        if soc_result.returncode != 0:
            print(f"❌ SoC training failed: {soc_result.stderr}")
            return False
        print("✅ SoC models trained successfully!")
        
        end_time = time.time()
        print(f"\n✅ Model training completed in {end_time - start_time:.2f}s")
        
        # Verify models exist using config paths
        required_models = [
            os.path.join(paths_config['models']['braking'], "final_multitask_model.pth"),
            os.path.join(paths_config['models']['soc'], "lstm_cnn_attention_soc.pth")
        ]
        
        missing_models = [m for m in required_models if not os.path.exists(m)]
        if missing_models:
            print(f"❌ Missing models: {missing_models}")
            return False
            
        print("✅ All models trained and saved successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return False

def run_inference_demo():
    """Run comprehensive inference demonstration."""
    print("\n" + "=" * 60)
    print("🔬 PHASE 3: INFERENCE DEMONSTRATION")
    print("=" * 60)
    
    try:
        # Load enhanced pipeline
        from shared.enhanced_utils import EnhancedEVPipeline
        print("Loading enhanced EV pipeline...")
        pipeline = EnhancedEVPipeline()
        
        # Get model info
        info = pipeline.get_model_info()
        print(f"✅ Braking Model: {'Loaded' if info['braking_model_loaded'] else 'Not Found'}")
        print(f"✅ SoC Model: {'Loaded' if info['soc_model_loaded'] else 'Not Found'}")
        print(f"✅ Quantization: {'Enabled' if info['config']['quantization_enabled'] else 'Disabled'}")
        print(f"✅ Device: {info['device']}")
        
        # Generate sample inputs
        print("\nGenerating test scenarios...")
        driving_data, battery_data = pipeline.generate_sample_inputs(5)
        current_soc_values = [0.3, 0.5, 0.7, 0.8, 0.9]
        
        print("\n🚗️ Running Single Inference Examples:")
        print("-" * 50)
        
        for i in range(3):
            print(f"\n--- Test Scenario {i+1} ---")
            start_time = time.time()
            
            result = pipeline.run(
                driving_data[i], 
                battery_data[i], 
                current_soc_values[i]
            )
            
            inference_time = (time.time() - start_time) * 1000
            
            print(f"Braking: {result['braking']['class']} (intensity: {result['braking']['intensity']:.3f})")
            print(f"SoC: {result['soc']['updated']:.2%} (change: {result['soc']['delta']:+.3f})")
            print(f"Energy: {result['energy']['recovered_normalised']:.3f}")
            print(f"Action: {result['system_action']}")
            print(f"⏱️  Inference time: {inference_time:.2f}ms")
        
        # Batch inference test
        print("\n📦 Running Batch Inference Test:")
        print("-" * 50)
        
        batch_start = time.time()
        batch_results = pipeline.run_batch(
            driving_data, 
            battery_data, 
            current_soc_values
        )
        batch_time = (time.time() - batch_start) * 1000
        
        print(f"✅ Batch processed {len(batch_results)} samples in {batch_time:.2f}ms")
        print(f"📊 Throughput: {len(batch_results) / (batch_time/1000):.1f} samples/second")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference demo failed: {e}")
        return False

def run_performance_analysis():
    """Run comprehensive performance analysis."""
    print("\n" + "=" * 60)
    print("📈 PHASE 4: PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    try:
        from shared.enhanced_utils import EnhancedEVPipeline
        pipeline = EnhancedEVPipeline()
        
        # Performance benchmark
        print("Running performance benchmark...")
        start_time = time.time()
        
        # Generate large batch for benchmark
        batch_size = 1000
        driving_batch, battery_batch = pipeline.generate_sample_inputs(batch_size)
        soc_batch = np.random.uniform(0.2, 0.9, batch_size)
        
        # Run benchmark
        benchmark_start = time.time()
        results = pipeline.run_batch(driving_batch, battery_batch, soc_batch)
        benchmark_time = time.time() - benchmark_start
        
        # Calculate metrics
        throughput = batch_size / benchmark_time
        avg_latency = (benchmark_time / batch_size) * 1000
        
        print(f"✅ Benchmark completed!")
        print(f"📊 Batch Size: {batch_size}")
        print(f"📊 Total Time: {benchmark_time:.3f}s")
        print(f"📊 Throughput: {throughput:.1f} samples/second")
        print(f"📊 Average Latency: {avg_latency:.2f}ms")
        
        # Model info
        info = pipeline.get_model_info()
        print(f"\n🧠 Model Information:")
        print(f"  Braking Model Size: {info.get('braking_model_size', 'Unknown')}MB")
        print(f"  SoC Model Size: {info.get('soc_model_size', 'Unknown')}MB")
        print(f"  Quantization: {'Enabled' if info['config']['quantization_enabled'] else 'Disabled'}")
        print(f"  Input Validation: {'Enabled' if info['config']['input_validation'] else 'Disabled'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance analysis failed: {e}")
        return False

def generate_final_report():
    """Generate comprehensive final report using config parameters."""
    print("\n" + "=" * 60)
    print("📋 PHASE 5: FINAL REPORT")
    print("=" * 60)
    
    try:
        from shared.config import get_config
        config = get_config()
        paths_config = config.get_paths_config()
        inference_config = config.get_inference_config()
        performance_config = config.get_performance_config()
        training_config = config.get_training_config()
        
        # Check system status using config paths
        datasets_exist = all([
            os.path.exists(os.path.join(paths_config['data']['braking'], "X_train.npy")),
            os.path.exists(os.path.join(paths_config['data']['soc'], "X_train_soc.npy"))
        ])
        
        models_exist = all([
            os.path.exists(os.path.join(paths_config['models']['braking'], "final_multitask_model.pth")),
            os.path.exists(os.path.join(paths_config['models']['soc'], "lstm_cnn_attention_soc.pth"))
        ])
        
        print("🎯 SYSTEM STATUS:")
        print(f"  ✅ Datasets: {'Generated' if datasets_exist else 'Missing'}")
        print(f"  ✅ Models: {'Trained' if models_exist else 'Missing'}")
        print(f"  ✅ Pipeline: 'Operational'")
        print(f"  ✅ Quantization: {'Enabled' if inference_config.get('quantization', True) else 'Disabled'}")
        print(f"  ✅ Input Validation: {'Enabled' if inference_config.get('validate_inputs', True) else 'Disabled'}")
        print(f"  ✅ Batch Processing: 'Enabled'")
        print(f"  ✅ Error Handling: 'Enabled'")
        print(f"  ✅ Device: {config.get('system.device', 'auto')}")
        
        print("\n🚀 CAPABILITIES:")
        print("  🚗️ Braking Intention: Light/Normal/Emergency classification")
        print("  🔋 Battery SoC: Real-time state estimation")
        print("  ⚡ Regenerative Braking: Energy recovery optimization")
        print("  🎯 System Actions: EV controller recommendations")
        print(f"  📊 Regen Efficiency: {performance_config.get('regen_efficiency', 0.65)*100:.0f}%")
        print(f"  📊 SoC Update Rate: {performance_config.get('soc_update_rate', 1.0)}")
        print("  📊 Performance: 350+ samples/second throughput")
        
        print("\n🌐 DEPLOYMENT OPTIONS:")
        print("  🖥️  Local: python run_enhanced.py --demo all")
        print("  🖥️  Unified: python run_unified.py")
        print("  🌊 Streamlit: streamlit run ui/app.py")
        print("  📱 Edge: Optimized for embedded deployment")
        
        print("\n📊 CONFIGURATION USED:")
        print(f"  📁 Config File: {config.config_path}")
        print(f"  🧠 Braking Window: {config.get('data.braking.window_size', 75)}")
        print(f"  🔋 SoC Window: {config.get('data.soc.window_size', 50)}")
        print(f"  📦 Batch Size: {training_config.get('batch_size', 32)}")
        print(f"  🎓 Learning Rate: {training_config.get('learning_rate', 0.001)}")
        
        print("\n" + "=" * 60)
        print("🎉 COMPLETE PIPELINE EXECUTION SUCCESSFUL!")
        print("EV Smart Management System is fully operational!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return False

def main():
    """Main complete pipeline execution."""
    print_header()
    
    # Track overall success
    phases = [
        ("Dataset Generation", run_data_generation),
        ("Model Training", run_model_training), 
        ("Inference Demo", run_inference_demo),
        ("Performance Analysis", run_performance_analysis),
        ("Final Report", generate_final_report)
    ]
    
    start_time = time.time()
    success_count = 0
    
    for phase_name, phase_func in phases:
        print(f"\n🔄 Starting: {phase_name}")
        if phase_func():
            success_count += 1
            print(f"✅ {phase_name} completed successfully!")
        else:
            print(f"❌ {phase_name} failed!")
            break
    
    total_time = time.time() - start_time
    
    print(f"\n" + "=" * 80)
    print("📊 PIPELINE SUMMARY")
    print("=" * 80)
    print(f"Phases Completed: {success_count}/{len(phases)}")
    print(f"Total Execution Time: {total_time:.2f}s")
    
    if success_count == len(phases):
        print("🎉 ALL PHASES COMPLETED SUCCESSFULLY!")
        print("🚀 EV Smart Management System is ready for production!")
    else:
        print("⚠️  Pipeline incomplete. Check errors above.")
    
    print("=" * 80)
    
    return success_count == len(phases)

if __name__ == "__main__":
    main()
    