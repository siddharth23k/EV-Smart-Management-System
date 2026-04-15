
# Loads raw NASA battery cycle CSVs, computes SOC labels via Coulomb Counting,
# creates sliding window sequences, and saves train/val/test .npy files.


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR    = os.path.join(os.path.dirname(__file__))          # modules/soc/data/
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__))
WINDOW_SIZE = 50       # timesteps per sample (sliding window)
STEP_SIZE   = 10       # stride between windows
CAPACITY_AH = 2.0      # nominal battery capacity in Ah (NASA 18650 cells)
MIN_ROWS    = WINDOW_SIZE + 5   # skip files that are too short

FEATURE_COLS = ["Voltage_measured", "Current_measured", "Temperature_measured"]
# SOC via Coulomb Counting
def compute_soc(current: np.ndarray, time: np.ndarray,
                capacity_ah: float = CAPACITY_AH) -> np.ndarray:
    """
    Estimates SOC using Coulomb Counting (ampere-hour integration).
    SOC(t) = SOC(0) - (1/Q) * cumulative_charge(t)

    For discharge cycles (current < 0), SOC starts at 1.0 and decreases.
    For charge cycles  (current > 0), SOC starts at 0.0 and increases.
    Values are clipped to [0, 1].
    """
    dt        = np.diff(time, prepend=time[0])      # time deltas (seconds)
    dt        = np.clip(dt, 0, 60)                  # guard against huge gaps
    charge_ah = current * dt / 3600.0               # As → Ah per timestep

    # Determine cycle type by dominant current sign
    is_discharge = np.mean(current) < 0

    if is_discharge:
        soc_0        = 1.0
        cumulative   = np.cumsum(charge_ah)         # negative values → SOC ↓
        soc          = soc_0 + cumulative / capacity_ah
    else:
        soc_0        = 0.0
        cumulative   = np.cumsum(charge_ah)
        soc          = soc_0 + cumulative / capacity_ah

    return np.clip(soc, 0.0, 1.0)


def sliding_windows(features: np.ndarray, soc: np.ndarray,
                    window: int, step: int):
    """
    Converts a time-series into overlapping windows.
    X shape : (N, window, n_features)
    y shape : (N,)  — SOC at the LAST timestep of each window
    """
    X, y = [], []
    for start in range(0, len(features) - window, step):
        end = start + window
        X.append(features[start:end])
        y.append(soc[end - 1])          # predict SOC at end of window
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def main():
    csv_files = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".csv") and f[0].isdigit()
    ])

    print(f"Found {len(csv_files)} CSV files in {DATA_DIR}")

    all_X, all_y = [], []
    skipped = 0

    for fname in csv_files:
        path = os.path.join(DATA_DIR, fname)
        try:
            df = pd.read_csv(path)
        except Exception:
            skipped += 1
            continue

        # Check required columns exist
        required = FEATURE_COLS + ["Current_measured", "Time"]
        if not all(c in df.columns for c in required):
            skipped += 1
            continue

        df = df.dropna(subset=required)

        if len(df) < MIN_ROWS:
            skipped += 1
            continue

        current = df["Current_measured"].values
        time    = df["Time"].values
        soc     = compute_soc(current, time)

        features = df[FEATURE_COLS].values

        # Normalize features per-cycle (zero mean, unit std)
        mean = features.mean(axis=0)
        std  = features.std(axis=0) + 1e-8
        features = (features - mean) / std

        X_win, y_win = sliding_windows(features, soc, WINDOW_SIZE, STEP_SIZE)

        if len(X_win) == 0:
            skipped += 1
            continue

        all_X.append(X_win)
        all_y.append(y_win)

    if len(all_X) == 0:
        raise RuntimeError("No valid windows extracted. Check DATA_DIR path.")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    print(f"Total windows : {len(X)}")
    print(f"Skipped files : {skipped}")
    print(f"X shape       : {X.shape}  (samples, timesteps, features)")
    print(f"y shape       : {y.shape}  (SOC values in [0,1])")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, shuffle=True
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    print(f"\nSplit sizes:")
    print(f"  Train : {len(X_train)}")
    print(f"  Val   : {len(X_val)}")
    print(f"  Test  : {len(X_test)}")

    np.save(os.path.join(OUTPUT_DIR, "X_train_soc.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_val_soc.npy"),   X_val)
    np.save(os.path.join(OUTPUT_DIR, "X_test_soc.npy"),  X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_train_soc.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "y_val_soc.npy"),   y_val)
    np.save(os.path.join(OUTPUT_DIR, "y_test_soc.npy"),  y_test)

    print(f"\nSaved .npy files to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()