"""
run_unified.py  (project root)
Entry point for the Unified EV Smart Management System.
Runs both modules together and shows integrated output.
"""

import numpy as np
from shared.utils import UnifiedEVPipeline

# Check if models exist, use dummy if not
import os
if not all(os.path.exists(p) for p in [
    "modules/braking/models/final_multitask_model.pth",
    "modules/soc/models/lstm_cnn_attention_soc.pth",
    "modules/braking/models/best_ga_hyperparams.json"
]):
    print("⚠️  Missing model weights. Using dummy predictions.")
    print("   Run: python modules/train/train_all_models.py")



def generate_sample_inputs():
    """Generate synthetic sample inputs for demo when real data isn't loaded."""
    # Driving window: (75, 3) — speed, acceleration, brake pedal
    # Simulates emergency braking scenario
    t = np.linspace(0, 1, 75)
    speed       = 60 - 30 * t + np.random.normal(0, 0.5, 75)
    accel       = -5 * t + np.random.normal(0, 0.1, 75)
    brake_pedal = 0.3 + 0.5 * t + np.random.normal(0, 0.02, 75)
    brake_pedal = np.clip(brake_pedal, 0, 1)
    driving_window = np.stack([speed, accel, brake_pedal], axis=1).astype(np.float32)

    # Normalize (same as training preprocessing)
    mean = driving_window.mean(axis=0)
    std  = driving_window.std(axis=0) + 1e-8
    driving_window = (driving_window - mean) / std

    # Battery window: (50, 3) — voltage, current, temperature
    voltage     = 3.8 - 0.5 * np.linspace(0, 1, 50) + np.random.normal(0, 0.01, 50)
    current     = -1.0 * np.ones(50) + np.random.normal(0, 0.05, 50)
    temperature = 25 + np.random.normal(0, 0.5, 50)
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
    print("Initialising Unified EV Smart Management System...")
    pipeline = UnifiedEVPipeline()

    print("\nGenerating sample inputs...")
    driving_window, battery_window = generate_sample_inputs()

    print(f"Driving window shape : {driving_window.shape}")
    print(f"Battery window shape : {battery_window.shape}")

    # Run unified inference
    result = pipeline.run(
        driving_window=driving_window,
        battery_window=battery_window,
        current_soc=0.65,   # assume 65% SOC currently
    )

    print_result(result)