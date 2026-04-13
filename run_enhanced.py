#!/usr/bin/env python3
"""
Enhanced EV Smart Management System
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from shared.config import get_config
from shared.enhanced_utils import EnhancedEVPipeline

def print_header():
    print("=" * 70)
    print("🚀 ENHANCED EV SMART MANAGEMENT SYSTEM")
    print("=" * 70)
    print("Features: Input Validation | Error Handling | Batch Inference | Quantization")
    print("=" * 70)

def print_model_info(pipeline):
    info = pipeline.get_model_info()
    print("\n📊 MODEL STATUS:")
    print(f"  Braking Model: {'✅ Loaded' if info['braking_model_loaded'] else '❌ Not Found'}")
    print(f"  SoC Model: {'✅ Loaded' if info['soc_model_loaded'] else '❌ Not Found'}")
    print(f"  Quantization: {'✅ Enabled' if info['config']['quantization_enabled'] else '❌ Disabled'}")
    print(f"  Input Validation: {'✅ Enabled' if info['config']['input_validation'] else '❌ Disabled'}")
    print(f"  Device: {info['device']}")

def run_single_inference_demo(pipeline):
    print("\n" + "=" * 70)
    print("🔬 SINGLE INFERENCE DEMONSTRATION")
    print("=" * 70)
    
    try:
        driving_window, battery_window = pipeline.generate_sample_inputs(1)
        current_soc = 0.65
        
        print(f"Input shapes: driving={driving_window.shape}, battery={battery_window.shape}")
        print(f"Current SoC: {current_soc:.2f}")
        
        start_time = time.time()
        result = pipeline.run(driving_window, battery_window, current_soc)
        end_time = time.time()
        
        print(f"\n⚡ Inference completed in {(end_time - start_time)*1000:.2f}ms")
        print("\n📋 RESULTS:")
        print(f"  Braking: {result['braking']['class']} (confidence: {result['braking']['confidence']:.2f})")
        print(f"  Intensity: {result['braking']['intensity']:.3f}")
        print(f"  Energy Recovered: {result['energy']['recovered_normalised']:.3f}")
        print(f"  SoC: {result['soc']['current']:.3f} → {result['soc']['updated']:.3f} (+{result['soc']['delta']:.3f})")
        print(f"  System Action: {result['system_action']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Single inference failed: {e}")
        return False

def run_batch_inference_demo(pipeline):
    print("\n" + "=" * 70)
    print("📦 BATCH INFERENCE DEMONSTRATION")
    print("=" * 70)
    
    try:
        batch_size = 5
        driving_windows, battery_windows = pipeline.generate_sample_inputs(batch_size)
        current_socs = [0.3, 0.5, 0.7, 0.9, 0.2]
        
        print(f"Batch size: {batch_size}")
        print(f"Input shapes: driving={len(driving_windows)} samples, battery={len(battery_windows)} samples")
        
        start_time = time.time()
        results = pipeline.run(driving_windows, battery_windows, current_socs)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / batch_size
        print(f"\n⚡ Batch inference completed in {total_time*1000:.2f}ms")
        print(f"   Average per sample: {avg_time*1000:.2f}ms")
        print(f"   Throughput: {batch_size/total_time:.1f} samples/second")
        
        print("\n📋 BATCH RESULTS:")
        for i, result in enumerate(results):
            print(f"  Sample {i+1}: {result['braking']['class']} | "
                  f"SoC {result['soc']['current']:.2f}→{result['soc']['updated']:.2f} | "
                  f"Energy {result['energy']['recovered_normalised']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch inference failed: {e}")
        return False

def run_validation_demo(pipeline):
    print("\n" + "=" * 70)
    print("🛡️  INPUT VALIDATION DEMONSTRATION")
    print("=" * 70)
    
    test_cases = [
        ("Valid input", lambda: (pipeline.generate_sample_inputs(1)[0], 0.65)),
        ("Invalid shape", lambda: (np.random.rand(10, 3), np.random.rand(50, 3), 0.65)),
        ("NaN values", lambda: (np.full((75, 3), np.nan), np.random.rand(50, 3), 0.65)),
        ("Invalid SoC", lambda: (pipeline.generate_sample_inputs(1)[0], 1.5)),
    ]
    
    for test_name, input_func in test_cases:
        try:
            print(f"\n🧪 Testing: {test_name}")
            inputs = input_func()
            if len(inputs) == 3:
                driving_window, battery_window, current_soc = inputs
            else:
                driving_window, battery_window = inputs
                current_soc = 1.5 if "Invalid SoC" in test_name else 0.65
            
            result = pipeline.run(driving_window, battery_window, current_soc)
            print(f"   ✅ Passed: {result['braking']['class']}")
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)[:80]}...")

def run_performance_benchmark(pipeline):
    print("\n" + "=" * 70)
    print("🏎️  PERFORMANCE BENCHMARK")
    print("=" * 70)
    
    try:
        num_samples = 100
        driving_windows, battery_windows = pipeline.generate_sample_inputs(num_samples)
        current_socs = [0.5] * num_samples
        
        for _ in range(10):
            pipeline.run(driving_windows[0], battery_windows[0], current_socs[0])
        
        times = []
        for i in range(num_samples):
            start_time = time.time()
            pipeline.run(driving_windows[i], battery_windows[i], current_socs[i])
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times) * 1000
        min_time = min(times) * 1000
        max_time = max(times) * 1000
        throughput = num_samples / sum(times)
        
        print(f"Samples: {num_samples}")
        print(f"Average time: {avg_time:.2f}ms")
        print(f"Min time: {min_time:.2f}ms")
        print(f"Max time: {max_time:.2f}ms")
        print(f"Throughput: {throughput:.1f} samples/second")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Enhanced EV Smart Management System")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--demo", choices=["single", "batch", "validation", "benchmark", "all"], 
                       default="all", help="Demo mode to run")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                       help="Device to use for inference")
    
    args = parser.parse_args()
    
    print_header()
    
    try:
        print("🔧 Initializing Enhanced EV Pipeline...")
        pipeline = EnhancedEVPipeline(args.config)
        
        if args.device != "auto":
            import torch
            device = torch.device(args.device)
            print(f"📱 Using specified device: {device}")
        
        print_model_info(pipeline)
        
        success_count = 0
        total_count = 0
        
        if args.demo in ["single", "all"]:
            total_count += 1
            if run_single_inference_demo(pipeline):
                success_count += 1
        
        if args.demo in ["batch", "all"]:
            total_count += 1
            if run_batch_inference_demo(pipeline):
                success_count += 1
        
        if args.demo in ["validation", "all"]:
            total_count += 1
            if run_validation_demo(pipeline):
                success_count += 1
        
        if args.demo in ["benchmark", "all"]:
            total_count += 1
            if run_performance_benchmark(pipeline):
                success_count += 1
        
        print("\n" + "=" * 70)
        print("📊 SUMMARY")
        print("=" * 70)
        print(f"Demos completed: {success_count}/{total_count}")
        
        if success_count == total_count:
            print("🎉 All demos completed successfully!")
            print("✅ Enhanced EV Smart Management System is fully operational!")
        else:
            print("⚠️  Some demos failed. Check the errors above.")
        
        return success_count == total_count
        
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        print("Please check:")
        print("  1. Configuration file exists and is valid")
        print("  2. Model files are trained and available")
        print("  3. Dependencies are installed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
