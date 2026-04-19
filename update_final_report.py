#!/usr/bin/env python3
"""
Update Final Report using existing trained models
Collects results from already completed training runs
"""

import os
import sys
import json
import time
from datetime import datetime
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def check_existing_results():
    """Check for existing training results."""
    print("=== CHECKING EXISTING RESULTS ===")
    
    results = {}
    
    # Check Multi-Objective GA results
    from shared.config import get_config
    config = get_config()
    paths = config.get_paths_config()
    ga_results_path = os.path.join(paths['models']['soc'], 'multi_objective_ga_results.json')
    if os.path.exists(ga_results_path):
        try:
            with open(ga_results_path, "r") as f:
                ga_data = json.load(f)
            
            # Extract key information
            best_individual = ga_data.get("best_individual", {})
            results["Multi-Objective GA"] = {
                "status": "success",
                "duration": ga_data.get("optimization_time", 0),
                "output": f"Best RMSE: {best_individual.get('rmse', 'N/A')}\n" +
                       f"Best Efficiency: {best_individual.get('efficiency', 'N/A')}ms\n" +
                       f"Robustness Loss: {best_individual.get('robustness_loss', 'N/A')}\n" +
                       f"Hyperparameters: {best_individual.get('hyperparameters', {})}"
            }
            print("  Multi-Objective GA: Results found")
        except Exception as e:
            results["Multi-Objective GA"] = {
                "status": "failed",
                "duration": 0,
                "error": f"Failed to read GA results: {str(e)}"
            }
            print(f"  Multi-Objective GA: Error reading results - {e}")
    else:
        results["Multi-Objective GA"] = {
            "status": "failed",
            "duration": 0,
            "error": "GA results file not found"
        }
        print("  Multi-Objective GA: Results file not found")
    
    # Check Adaptive Ensemble results
    ensemble_results_path = "modules/soc/models/adaptive_ensemble_results.json"
    if os.path.exists(ensemble_results_path):
        try:
            with open(ensemble_results_path, "r") as f:
                ensemble_data = json.load(f)
            
            final_eval = ensemble_data.get("final_evaluation", {})
            results["Adaptive Ensemble"] = {
                "status": "success",
                "duration": 0,  # Duration not tracked in individual results
                "output": f"Final Ensemble RMSE: {final_eval.get('ensemble_rmse', 'N/A')}\n" +
                       f"Individual Model RMSEs:\n" +
                       f"  LSTM-CNN: {final_eval.get('lstm_rmse', 'N/A')}\n" +
                       f"  Transformer: {final_eval.get('transformer_rmse', 'N/A')}\n" +
                       f"  Physics: {final_eval.get('physics_rmse', 'N/A')}\n" +
                       f"Optimized Weights: {final_eval.get('weights', 'N/A')}"
            }
            print("  Adaptive Ensemble: Results found")
        except Exception as e:
            results["Adaptive Ensemble"] = {
                "status": "failed",
                "duration": 0,
                "error": f"Failed to read ensemble results: {str(e)}"
            }
            print(f"  Adaptive Ensemble: Error reading results - {e}")
    else:
        # Check if ensemble model exists as fallback
        ensemble_model_path = "modules/soc/models/adaptive_ensemble.json"
        if os.path.exists(ensemble_model_path):
            try:
                with open(ensemble_model_path, "r") as f:
                    ensemble_config = json.load(f)
                
                results["Adaptive Ensemble"] = {
                    "status": "success",
                    "duration": 0,
                    "output": f"Ensemble model configuration found\n" +
                           f"Weights: {ensemble_config.get('weights', 'N/A')}\n" +
                           f"Adaptation rate: {ensemble_config.get('adaptation_rate', 'N/A')}"
                }
                print("  Adaptive Ensemble: Model configuration found")
            except Exception as e:
                results["Adaptive Ensemble"] = {
                    "status": "failed",
                    "duration": 0,
                    "error": f"Failed to read ensemble config: {str(e)}"
                }
                print(f"  Adaptive Ensemble: Error reading config - {e}")
        else:
            results["Adaptive Ensemble"] = {
                "status": "failed",
                "duration": 0,
                "error": "Ensemble results and configuration not found"
            }
            print("  Adaptive Ensemble: No results or configuration found")
    
    return results

def generate_final_report(results):
    """Generate final report from existing results."""
    print("=== GENERATING FINAL REPORT ===")
    
    # Count successful and failed components
    components_trained = sum(1 for r in results.values() if r["status"] == "success")
    components_failed = sum(1 for r in results.values() if r["status"] == "failed")
    
    # Use current time for the report
    now = datetime.now()
    
    # Create report
    report = {
        "training_summary": {
            "total_duration": sum(r["duration"] for r in results.values()),
            "start_time": now.isoformat(),  # Using current time as we don't have original start time
            "end_time": now.isoformat(),
            "components_trained": components_trained,
            "components_failed": components_failed,
            "note": "Report generated from existing training results"
        },
        "component_results": results,
        "recommendations": []
    }
    
    # Add recommendations for failed components
    for component, result in results.items():
        if result["status"] == "failed":
            report["recommendations"].append(f"{component}: Check error logs and fix issues")
    
    # Ensure training_checkpoints directory exists
    os.makedirs("training_checkpoints", exist_ok=True)
    
    # Save report
    report_path = "training_checkpoints/final_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    
    print(f"Final report updated: {report_path}")
    return report_path

def main():
    """Update final report from existing results."""
    print("EV SMART MANAGEMENT SYSTEM - UPDATE FINAL REPORT")
    print("=" * 60)
    
    # Check existing results
    results = check_existing_results()
    
    # Generate final report
    report_path = generate_final_report(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("FINAL REPORT UPDATE SUMMARY")
    print("=" * 60)
    
    for component, result in results.items():
        status = "SUCCESS" if result["status"] == "success" else "FAILED"
        print(f"{component}: {status}")
        
        if result["status"] == "failed":
            print(f"  Error: {result['error']}")
    
    total_components = len(results)
    successful = sum(1 for r in results.values() if r["status"] == "success")
    
    print(f"\nOverall: {successful}/{total_components} components successful")
    print(f"Report updated: {report_path}")
    
    return successful > 0  # Success if at least one component worked

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
