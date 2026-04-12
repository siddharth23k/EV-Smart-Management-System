#!/usr/bin/env python3
"""
Complete Training Pipeline for EV Smart Management System
Trains all models with full parameters and optimizations
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))


class CompleteTrainingPipeline:
    """Complete training pipeline with progress tracking."""
    
    def __init__(self):
        self.start_time = time.time()
        self.results = {}
        self.checkpoint_dir = "training_checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def log_progress(self, component: str, status: str, details: str = ""):
        """Log training progress."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {component}: {status}"
        if details:
            log_entry += f" - {details}"
        
        print(log_entry)
        
        # Save to log file
        with open(f"{self.checkpoint_dir}/training_log.txt", "a") as f:
            f.write(log_entry + "\n")
    
    def run_command(self, command: str, component: str, timeout: int = 7200):
        """Run training command with timeout and error handling."""
        self.log_progress(component, "Starting", f"Command: {command}")
        
        try:
            start = time.time()
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=project_root
            )
            
            duration = time.time() - start
            
            if result.returncode == 0:
                self.log_progress(component, "SUCCESS", f"Duration: {duration:.1f}s")
                self.results[component] = {
                    "status": "success",
                    "duration": duration,
                    "output": result.stdout[-500:]  # Last 500 chars
                }
                return True
            else:
                self.log_progress(component, "FAILED", f"Error: {result.stderr[-200:]}")
                self.results[component] = {
                    "status": "failed",
                    "duration": duration,
                    "error": result.stderr[-500:]
                }
                return False
                
        except subprocess.TimeoutExpired:
            self.log_progress(component, "TIMEOUT", f"Timeout after {timeout}s")
            self.results[component] = {
                "status": "timeout",
                "duration": timeout,
                "error": "Training timeout"
            }
            return False
        except Exception as e:
            self.log_progress(component, "ERROR", f"Exception: {str(e)}")
            self.results[component] = {
                "status": "error",
                "duration": 0,
                "error": str(e)
            }
            return False
    
    def save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "total_duration": time.time() - self.start_time,
            "results": self.results
        }
        
        with open(f"{self.checkpoint_dir}/checkpoint.json", "w") as f:
            json.dump(checkpoint, f, indent=4)
    
    def phase1_foundation_models(self):
        """Phase 1: Train foundation models with full parameters."""
        self.log_progress("PHASE 1", "Starting Foundation Models Training")
        
        # 1.1 Multi-Objective GA Optimizer
        success = self.run_command(
            "python modules/soc/models/multi_objective_ga_optimizer.py",
            "Multi-Objective GA",
            timeout=10800  # 3 hours
        )
        
        if not success:
            self.log_progress("PHASE 1", "STOPPED", "Multi-Objective GA failed")
            return False
        
        self.save_checkpoint()
        
        # 1.2 Adaptive Ensemble (full version)
        success = self.run_command(
            "python modules/soc/models/adaptive_ensemble.py",
            "Adaptive Ensemble",
            timeout=7200  # 2 hours
        )
        
        if not success:
            self.log_progress("PHASE 1", "STOPPED", "Adaptive Ensemble failed")
            return False
        
        self.save_checkpoint()
        
        # 1.3 Physics-Informed Model
        success = self.run_command(
            "python modules/soc/models/physics_informed_soc.py",
            "Physics-Informed Model",
            timeout=3600  # 1 hour
        )
        
        if not success:
            self.log_progress("PHASE 1", "STOPPED", "Physics-Informed Model failed")
            return False
        
        self.save_checkpoint()
        self.log_progress("PHASE 1", "COMPLETED", "All foundation models trained")
        return True
    
    def phase2_integration(self):
        """Phase 2: Integration and pipeline testing."""
        self.log_progress("PHASE 2", "Starting Integration")
        
        # 2.1 Test unified pipeline with all models
        success = self.run_command(
            "python run_unified.py",
            "Unified Pipeline Test",
            timeout=300  # 5 minutes
        )
        
        if not success:
            self.log_progress("PHASE 2", "FAILED", "Unified pipeline test failed")
            return False
        
        self.save_checkpoint()
        self.log_progress("PHASE 2", "COMPLETED", "Integration successful")
        return True
    
    def phase3_cognitive_system(self):
        """Phase 3: Cognitive system training."""
        self.log_progress("PHASE 3", "Starting Cognitive System")
        
        success = self.run_command(
            "python shared/cognitive_manager.py",
            "Cognitive Manager",
            timeout=1800  # 30 minutes
        )
        
        if not success:
            self.log_progress("PHASE 3", "FAILED", "Cognitive system failed")
            return False
        
        self.save_checkpoint()
        self.log_progress("PHASE 3", "COMPLETED", "Cognitive system trained")
        return True
    
    def phase4_benchmarking(self):
        """Phase 4: Performance benchmarking."""
        self.log_progress("PHASE 4", "Starting Benchmarking")
        
        # Run comprehensive test system
        success = self.run_command(
            "python test_system.py",
            "Comprehensive Testing",
            timeout=600  # 10 minutes
        )
        
        if not success:
            self.log_progress("PHASE 4", "FAILED", "Benchmarking failed")
            return False
        
        self.save_checkpoint()
        self.log_progress("PHASE 4", "COMPLETED", "Benchmarking successful")
        return True
    
    def generate_report(self):
        """Generate final training report."""
        total_duration = time.time() - self.start_time
        
        report = {
            "training_summary": {
                "total_duration": total_duration,
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": datetime.now().isoformat(),
                "components_trained": len([r for r in self.results.values() if r["status"] == "success"]),
                "components_failed": len([r for r in self.results.values() if r["status"] != "success"])
            },
            "component_results": self.results,
            "recommendations": self._generate_recommendations()
        }
        
        with open(f"{self.checkpoint_dir}/final_report.json", "w") as f:
            json.dump(report, f, indent=4)
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING PIPELINE SUMMARY")
        print("="*60)
        print(f"Total Duration: {total_duration/3600:.1f} hours")
        print(f"Components Trained: {report['training_summary']['components_trained']}")
        print(f"Components Failed: {report['training_summary']['components_failed']}")
        
        for component, result in self.results.items():
            status_icon = "SUCCESS" if result["status"] == "success" else "FAILED"
            print(f"  {component}: {status_icon} ({result['duration']/60:.1f} min)")
        
        print("="*60)
        
        return report
    
    def _generate_recommendations(self):
        """Generate recommendations based on results."""
        recommendations = []
        
        for component, result in self.results.items():
            if result["status"] == "timeout":
                recommendations.append(f"{component}: Consider reducing parameters or increasing timeout")
            elif result["status"] == "failed":
                recommendations.append(f"{component}: Check error logs and fix issues")
        
        if not recommendations:
            recommendations.append("All components trained successfully. System ready for production.")
        
        return recommendations
    
    def run_complete_pipeline(self):
        """Run the complete training pipeline."""
        print("Starting Complete Training Pipeline for EV Smart Management System")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Phase 1: Foundation Models (Most Critical)
        if not self.phase1_foundation_models():
            print("Phase 1 failed. Stopping pipeline.")
            return self.generate_report()
        
        # Phase 2: Integration
        if not self.phase2_integration():
            print("Phase 2 failed. Foundation models trained but integration failed.")
            return self.generate_report()
        
        # Phase 3: Cognitive System (Optional but recommended)
        if not self.phase3_cognitive_system():
            print("Phase 3 failed. Foundation models work but cognitive system failed.")
            # Continue anyway since cognitive system is optional
        
        # Phase 4: Benchmarking
        if not self.phase4_benchmarking():
            print("Phase 4 failed. Models trained but benchmarking failed.")
            # Continue anyway
        
        return self.generate_report()


def main():
    """Main function."""
    pipeline = CompleteTrainingPipeline()
    
    # Check if user wants to run specific phase
    if len(sys.argv) > 1:
        phase = sys.argv[1]
        
        if phase == "phase1":
            pipeline.phase1_foundation_models()
        elif phase == "phase2":
            pipeline.phase2_integration()
        elif phase == "phase3":
            pipeline.phase3_cognitive_system()
        elif phase == "phase4":
            pipeline.phase4_benchmarking()
        else:
            print(f"Unknown phase: {phase}")
            print("Available phases: phase1, phase2, phase3, phase4")
    else:
        # Run complete pipeline
        report = pipeline.run_complete_pipeline()
        
        # Save final report
        with open("training_complete_report.json", "w") as f:
            json.dump(report, f, indent=4)
        
        print(f"\nFinal report saved to: training_complete_report.json")
        print(f"Training logs saved to: {pipeline.checkpoint_dir}/")


if __name__ == "__main__":
    main()
