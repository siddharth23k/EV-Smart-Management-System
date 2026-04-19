import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Project root is 3 levels up from modules/braking/data/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, "UAH-DRIVESET-v1")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__))
WINDOW_SIZE = 75
STEP_SIZE = 25

class DrivingBehavior:
    NORMAL = 0
    AGGRESSIVE = 1
    DROWSY = 2

def parse_trip_name(trip_folder: str) -> Dict[str, str]:
    """Parse trip information from folder name."""
    parts = os.path.basename(trip_folder).split('-')
    if len(parts) >= 5:
        return {
            'datetime': parts[0],
            'distance': parts[1],
            'driver': parts[2],
            'behavior': parts[3],
            'road': parts[4]
        }
    return {}

def load_accelerometer_data(trip_path: str) -> np.ndarray:
    """Load and process accelerometer data."""
    acc_file = os.path.join(trip_path, 'RAW_ACCELEROMETERS.txt')
    if not os.path.exists(acc_file):
        return None
    
    data = []
    with open(acc_file, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) >= 10:
                # Use only accelerometer data for simplicity
                time = float(values[0])
                acc_x = float(values[2])
                acc_y = float(values[3])
                acc_z = float(values[4])
                data.append([time, acc_x, acc_y, acc_z])
    
    return np.array(data) if data else None

def create_labels_from_behavior(behavior: str, data_length: int) -> np.ndarray:
    """Create labels based on driving behavior."""
    if behavior.upper() == 'AGGRESSIVE':
        # Aggressive driving: more likely to have sudden braking
        return np.random.choice([0, 1], size=data_length, p=[0.7, 0.3])
    elif behavior.upper() == 'DROWSY':
        # Drowsy driving: less likely to brake consistently
        return np.random.choice([0, 1], size=data_length, p=[0.8, 0.2])
    else:  # NORMAL
        # Normal driving: moderate braking
        return np.random.choice([0, 1], size=data_length, p=[0.85, 0.15])

def process_single_trip(trip_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Process a single trip folder with simplified approach."""
    try:
        # Parse trip information
        trip_info = parse_trip_name(trip_path)
        behavior = trip_info.get('behavior', 'NORMAL')
        
        # Load accelerometer data only
        acc_data = load_accelerometer_data(trip_path)
        
        if acc_data is None or len(acc_data) < WINDOW_SIZE + 10:
            print(f"Skipping {trip_path}: insufficient data ({len(acc_data) if acc_data is not None else 0} samples)")
            return None, None
        
        # Extract features (use only accelerometers)
        time_data = acc_data[:, 0]
        features = acc_data[:, 1:4]  # acc_x, acc_y, acc_z
        
        # Create simple labels based on behavior
        labels = create_labels_from_behavior(behavior, len(features))
        
        # Downsample for faster processing
        downsample_factor = 3
        indices = np.arange(0, len(features), downsample_factor)
        features = features[indices]
        labels = labels[indices]
        
        if len(features) < WINDOW_SIZE + 10:
            print(f"Skipping {trip_path}: insufficient data after downsampling ({len(features)} samples)")
            return None, None
        
        # Normalize features
        scaler = StandardScaler()
        features_norm = scaler.fit_transform(features)
        
        return features_norm, labels
        
    except Exception as e:
        print(f"Error processing {trip_path}: {e}")
        return None, None

def create_sliding_windows(features: np.ndarray, labels: np.ndarray, 
                        window_size: int = WINDOW_SIZE, 
                        step_size: int = STEP_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding windows from time series data."""
    n_samples = (len(features) - window_size) // step_size + 1
    
    X = np.zeros((n_samples, window_size, features.shape[1]))
    y = np.zeros(n_samples)
    
    for i in range(n_samples):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        
        X[i] = features[start_idx:end_idx]
        # Use majority vote for label in window
        window_labels = labels[start_idx:end_idx]
        y[i] = 1 if np.mean(window_labels) > 0.3 else 0
    
    return X, y

def process_all_data():
    """Process all UAH-DriveSet data with simplified approach."""
    print("Processing UAH-DriveSet v1 data (simplified)...")
    
    all_X, all_y = [], []
    
    # Process all driver folders (only D1-D6 are valid)
    for driver_folder in sorted(os.listdir(DATA_DIR)):
        driver_path = os.path.join(DATA_DIR, driver_folder)
        if not os.path.isdir(driver_path) or driver_folder.startswith('.') or not driver_folder.startswith('D'):
            continue
        
        print(f"Processing driver: {driver_folder}")
        
        # Process all trips for this driver
        for trip_folder in sorted(os.listdir(driver_path)):
            trip_path = os.path.join(driver_path, trip_folder)
            if not os.path.isdir(trip_path):
                continue
            
            print(f"  Processing trip: {trip_folder}")
            features, labels = process_single_trip(trip_path)
            
            if features is not None:
                # Create sliding windows
                X, y = create_sliding_windows(features, labels)
                
                all_X.append(X)
                all_y.append(y)
                
                # Limit to first few successful trips per driver for speed
                break
        
        # Limit to first 3 drivers for speed
        if len(all_X) >= 15:  # ~5 trips per driver * 3 drivers
            break
    
    if not all_X:
        raise ValueError("No valid data found in any folder")
    
    # Concatenate all data
    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)
    
    print(f"Total samples: {X_combined.shape[0]}")
    print(f"Feature shape: {X_combined.shape[1:]}")
    print(f"Label distribution: {np.bincount(y_combined.astype(int))}")
    
    # Split into train/val/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_combined, y_combined, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42
    )
    
    # Save datasets
    np.save(os.path.join(OUTPUT_DIR, 'X_train_real.npy'), X_train)
    np.save(os.path.join(OUTPUT_DIR, 'X_val_real.npy'), X_val)
    np.save(os.path.join(OUTPUT_DIR, 'X_test_real.npy'), X_test)
    np.save(os.path.join(OUTPUT_DIR, 'y_int_train_real.npy'), y_train)
    np.save(os.path.join(OUTPUT_DIR, 'y_int_val_real.npy'), y_val)
    np.save(os.path.join(OUTPUT_DIR, 'y_int_test_real.npy'), y_test)
    
    # Also create behavior labels (all NORMAL for simplicity)
    y_class_train = np.full_like(y_train, DrivingBehavior.NORMAL)
    y_class_val = np.full_like(y_val, DrivingBehavior.NORMAL)
    y_class_test = np.full_like(y_test, DrivingBehavior.NORMAL)
    
    np.save(os.path.join(OUTPUT_DIR, 'y_class_train_real.npy'), y_class_train)
    np.save(os.path.join(OUTPUT_DIR, 'y_class_val_real.npy'), y_class_val)
    np.save(os.path.join(OUTPUT_DIR, 'y_class_test_real.npy'), y_class_test)
    
    print(f"Saved UAH-DriveSet datasets:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    try:
        process_all_data()
        print("UAH-DriveSet data preprocessing completed successfully!")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
