import numpy as np
import json
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class DrivingScenario(Enum):
    # Different driving scenarios for simulation
    URBAN = "urban"
    HIGHWAY = "highway"
    AGGRESSIVE = "aggressive"
    ECO = "eco"
    EMERGENCY = "emergency"


@dataclass
class VehicleParameters:
    # EV vehicle physical parameters
    mass: float = 1800.0  # kg
    max_regen_power: float = 60000.0  # W (60kW)
    max_mechanical_brake: float = 8000.0  # N
    frontal_area: float = 2.5  # m^2
    drag_coefficient: float = 0.28
    rolling_resistance: float = 0.015
    wheel_radius: float = 0.35  # m
    motor_efficiency: float = 0.95
    regen_efficiency_curve: Dict[float, float] = None
    
    def __post_init__(self):
        if self.regen_efficiency_curve is None:
            # Regenerative braking efficiency vs speed (m/s)
            self.regen_efficiency_curve = {
                0.0: 0.0,    # No regen at standstill
                2.78: 0.3,   # 10 km/h: 30% efficiency
                5.56: 0.5,   # 20 km/h: 50% efficiency
                8.33: 0.65,  # 30 km/h: 65% efficiency
                11.11: 0.75, # 40 km/h: 75% efficiency
                13.89: 0.8,  # 50 km/h: 80% efficiency
                16.67: 0.82, # 60 km/h: 82% efficiency
                19.44: 0.8,  # 70 km/h: 80% efficiency
                22.22: 0.75, # 80 km/h: 75% efficiency
                25.0: 0.7,   # 90 km/h: 70% efficiency
                27.78: 0.65, # 100 km/h: 65% efficiency
                30.56: 0.6   # 110 km/h: 60% efficiency
            }


@dataclass
class RoadConditions:
    # Road and environmental conditions
    slope: float = 0.0  # degrees (positive = uphill)
    friction_coefficient: float = 0.8  # Dry asphalt
    wind_speed: float = 0.0  # m/s (headwind positive)
    air_density: float = 1.225  # kg/m^3


class RealisticEVSimulator:
    # Physics-based EV braking simulator
    
    def __init__(self, 
                 vehicle_params: Optional[VehicleParameters] = None,
                 road_conditions: Optional[RoadConditions] = None):
        self.vehicle = vehicle_params or VehicleParameters()
        self.road = road_conditions or RoadConditions()
        self.dt = 0.1  # Time step in seconds (100ms)
        self.seq_len = 75  # 7.5 seconds of data
        
    def _get_regen_efficiency(self, speed: float) -> float:
        #Get regenerative braking efficiency at given speed.
        speeds = sorted(self.vehicle.regen_efficiency_curve.keys())
        
        if speed <= speeds[0]:
            return self.vehicle.regen_efficiency_curve[speeds[0]]
        if speed >= speeds[-1]:
            return self.vehicle.regen_efficiency_curve[speeds[-1]]
        
        # Linear interpolation
        for i in range(len(speeds) - 1):
            if speeds[i] <= speed <= speeds[i + 1]:
                v1, v2 = speeds[i], speeds[i + 1]
                e1, e2 = self.vehicle.regen_efficiency_curve[v1], self.vehicle.regen_efficiency_curve[v2]
                return e1 + (e2 - e1) * (speed - v1) / (v2 - v1)
        
        return 0.65  # Default efficiency
    
    def _calculate_resistive_forces(self, speed: float) -> float:
        #Calculate total resistive force acting on vehicle.
        # Aerodynamic drag: F = 0.5 * ρ * Cd * A * v^2
        relative_wind_speed = speed + self.road.wind_speed
        aero_drag = 0.5 * self.road.air_density * self.vehicle.drag_coefficient * \
                    self.vehicle.frontal_area * relative_wind_speed**2
        
        # Rolling resistance: F = Cr * m * g * cos(θ)
        rolling_resistance = self.vehicle.rolling_resistance * self.vehicle.mass * \
                            9.81 * np.cos(np.radians(self.road.slope))
        
        # Gravitational component: F = m * g * sin(θ)
        gravity_component = self.vehicle.mass * 9.81 * np.sin(np.radians(self.road.slope))
        
        return aero_drag + rolling_resistance + gravity_component
    
    def _calculate_braking_force(self, brake_pedal: float, speed: float) -> Tuple[float, float]:
        #Calculate regenerative and mechanical braking forces.
        # Total desired braking force based on pedal position
        max_total_force = self.vehicle.max_mechanical_brake + \
                         (self.vehicle.max_regen_power / max(speed, 0.1))
        desired_total_force = brake_pedal * max_total_force
        
        # Regenerative braking limited by motor power and efficiency
        regen_efficiency = self._get_regen_efficiency(speed)
        max_regen_force = min(
            self.vehicle.max_regen_power / max(speed, 0.1),
            desired_total_force * regen_efficiency
        )
        
        regen_force = min(max_regen_force, desired_total_force)
        mechanical_force = max(0, desired_total_force - regen_force)
        
        return regen_force, mechanical_force
    
    def _simulate_driver_behavior(self, scenario: DrivingScenario, 
                                 initial_speed: float) -> Tuple[float, float, float]:
        # Generate realistic driver behavior patterns
        if scenario == DrivingScenario.URBAN:
            # Urban driving: frequent light to moderate braking
            target_brake = np.random.beta(2, 5) * 0.6
            speed_variation = np.random.normal(0, 2)
            
        elif scenario == DrivingScenario.HIGHWAY:
            # Highway: gentle braking, higher speeds
            target_brake = np.random.beta(1, 8) * 0.4
            speed_variation = np.random.normal(0, 1)
            
        elif scenario == DrivingScenario.AGGRESSIVE:
            # Aggressive: hard braking, rapid deceleration
            target_brake = np.random.beta(1, 2) * 0.9
            speed_variation = np.random.normal(0, 3)
            
        elif scenario == DrivingScenario.ECO:
            # Eco driving: gentle braking, maximizing regen
            target_brake = np.random.beta(3, 7) * 0.5
            speed_variation = np.random.normal(0, 1.5)
            
        else:  # EMERGENCY
            # Emergency: maximum braking
            target_brake = np.random.uniform(0.8, 1.0)
            speed_variation = np.random.normal(0, 5)
        
        # Add realistic constraints
        target_brake = np.clip(target_brake, 0, 1)
        final_speed = max(0, initial_speed + speed_variation)
        
        return target_brake, final_speed, speed_variation
    
    def generate_realistic_sample(self, 
                                 scenario: Optional[DrivingScenario] = None,
                                 initial_speed: Optional[float] = None,
                                 road_slope: Optional[float] = None) -> Tuple[np.ndarray, int, float]:
        #Generate a single realistic braking sample.
        # Randomize parameters if not provided
        if scenario is None:
            scenario = np.random.choice(list(DrivingScenario))
        if initial_speed is None:
            # Speed ranges based on scenario
            speed_ranges = {
                DrivingScenario.URBAN: (20, 60),
                DrivingScenario.HIGHWAY: (60, 120),
                DrivingScenario.AGGRESSIVE: (40, 100),
                DrivingScenario.ECO: (30, 80),
                DrivingScenario.EMERGENCY: (50, 130)
            }
            min_speed, max_speed = speed_ranges[scenario]
            initial_speed = np.random.uniform(min_speed, max_speed)
        if road_slope is not None:
            self.road.slope = road_slope
        
        # Get driver behavior
        target_brake, final_speed_target, _ = self._simulate_driver_behavior(scenario, initial_speed)
        
        # Determine braking class from target intensity
        if target_brake < 0.35:
            braking_class = 0  # Light Braking
        elif target_brake < 0.7:
            braking_class = 1  # Normal Braking
        else:
            braking_class = 2  # Emergency Braking
        
        # Simulate vehicle dynamics
        speed = initial_speed
        brake_pedal = 0.0
        
        trajectory = []
        
        for t in range(self.seq_len):
            # Gradual brake application (human driver behavior)
            brake_rate = np.random.uniform(0.02, 0.08)
            if scenario == DrivingScenario.EMERGENCY:
                brake_rate *= 2  # Faster brake application in emergency
            brake_pedal += (target_brake - brake_pedal) * brake_rate
            
            # Add driver inconsistency and sensor noise
            brake_pedal += np.random.normal(0, 0.02)
            brake_pedal = np.clip(brake_pedal, 0, 1)
            
            # Calculate forces
            regen_force, mechanical_force = self._calculate_braking_force(brake_pedal, speed)
            total_brake_force = regen_force + mechanical_force
            resistive_force = self._calculate_resistive_forces(speed)
            
            # Calculate acceleration (F = ma, negative for braking)
            acceleration = -(total_brake_force + resistive_force) / self.vehicle.mass
            
            # Add road noise and measurement errors
            acceleration += np.random.normal(0, 0.1)
            
            # Update speed
            speed += acceleration * self.dt
            speed = max(0, speed)  # Can't go backwards
            
            # Store trajectory [speed, acceleration, brake_pedal]
            trajectory.append([speed, acceleration, brake_pedal])
        
        return np.array(trajectory, dtype=np.float32), braking_class, target_brake
    
    def generate_dataset(self, 
                        n_samples: int = 15000,
                        scenario_distribution: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        #Generate complete dataset with scenario distribution.#
        if scenario_distribution is None:
            # Realistic distribution of driving scenarios
            scenario_distribution = {
                DrivingScenario.URBAN: 0.4,
                DrivingScenario.HIGHWAY: 0.3,
                DrivingScenario.AGGRESSIVE: 0.15,
                DrivingScenario.ECO: 0.1,
                DrivingScenario.EMERGENCY: 0.05
            }
        
        X, y_class, y_intensity = [], [], []
        
        # Generate samples according to distribution
        scenarios = list(scenario_distribution.keys())
        probabilities = list(scenario_distribution.values())
        
        for i in range(n_samples):
            # Choose scenario based on distribution
            scenario = np.random.choice(scenarios, p=probabilities)
            
            # Add variety to road conditions
            if np.random.random() < 0.3:  # 30% chance of slope
                road_slope = np.random.uniform(-5, 5)  # -5 to +5 degrees
            else:
                road_slope = 0.0
            
            # Generate sample
            x_i, yc_i, yi_i = self.generate_realistic_sample(
                scenario=scenario,
                road_slope=road_slope
            )
            
            X.append(x_i)
            y_class.append(yc_i)
            y_intensity.append(yi_i)
            
            if (i + 1) % 1000 == 0:
                print(f"Generated {i + 1}/{n_samples} samples...")
        
        return np.array(X), np.array(y_class), np.array(y_intensity)
    
    def save_simulation_metadata(self, filepath: str):
        #Save simulation parameters for reproducibility.
        metadata = {
            "vehicle_parameters": {
                "mass": self.vehicle.mass,
                "max_regen_power": self.vehicle.max_regen_power,
                "max_mechanical_brake": self.vehicle.max_mechanical_brake,
                "frontal_area": self.vehicle.frontal_area,
                "drag_coefficient": self.vehicle.drag_coefficient,
                "rolling_resistance": self.vehicle.rolling_resistance,
                "wheel_radius": self.vehicle.wheel_radius,
                "motor_efficiency": self.vehicle.motor_efficiency,
                "regen_efficiency_curve": self.vehicle.regen_efficiency_curve
            },
            "road_conditions": {
                "slope": self.road.slope,
                "friction_coefficient": self.road.friction_coefficient,
                "wind_speed": self.road.wind_speed,
                "air_density": self.road.air_density
            },
            "simulation_parameters": {
                "dt": self.dt,
                "seq_len": self.seq_len
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)


def generate_realistic_dataset(n_samples: int = 15000, 
                              save_path: str = "modules/braking/data") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #Main function to generate realistic EV braking dataset.
    print("Generating Realistic EV Braking Dataset")
    
    # Initialize simulator
    simulator = RealisticEVSimulator()
    
    # Generate dataset
    X, y_class, y_intensity = simulator.generate_dataset(n_samples=n_samples)
    
    # Shuffle dataset
    idx = np.random.permutation(len(X))
    X = X[idx]
    y_class = y_class[idx]
    y_intensity = y_intensity[idx]
    
    # Split dataset (70/15/15)
    n = len(X)
    n_train = int(0.7 * n)
    n_val = int(0.85 * n)
    
    X_train, X_val, X_test = X[:n_train], X[n_train:n_val], X[n_val:]
    y_class_train, y_class_val, y_class_test = y_class[:n_train], y_class[n_train:n_val], y_class[n_val:]
    y_int_train, y_int_val, y_int_test = y_intensity[:n_train], y_intensity[n_train:n_val], y_intensity[n_val:]
    
    # Save datasets
    import os
    os.makedirs(save_path, exist_ok=True)
    
    np.save(f"{save_path}/X_train_realistic.npy", X_train)
    np.save(f"{save_path}/X_val_realistic.npy", X_val)
    np.save(f"{save_path}/X_test_realistic.npy", X_test)
    
    np.save(f"{save_path}/y_class_train_realistic.npy", y_class_train)
    np.save(f"{save_path}/y_class_val_realistic.npy", y_class_val)
    np.save(f"{save_path}/y_class_test_realistic.npy", y_class_test)
    
    np.save(f"{save_path}/y_int_train_realistic.npy", y_int_train)
    np.save(f"{save_path}/y_int_val_realistic.npy", y_int_val)
    np.save(f"{save_path}/y_int_test_realistic.npy", y_int_test)
    
    # Save metadata
    simulator.save_simulation_metadata(f"{save_path}/realistic_simulation_metadata.json")
    
    print(f"\n Dataset Statistics ")
    print(f"Total samples: {n}")
    print(f"Training: {n_train} ({n_train/n*100:.1f}%)")
    print(f"Validation: {n_val-n_train} ({(n_val-n_train)/n*100:.1f}%)")
    print(f"Test: {n-n_val} ({(n-n_val)/n*100:.1f}%)")
    
    print(f"\n Class Distribution ")
    class_names = ["Light Braking", "Normal Braking", "Emergency Braking"]
    for i, class_name in enumerate(class_names):
        count = np.sum(y_class_train == i)
        print(f"{class_name}: {count} ({count/len(y_class_train)*100:.1f}%)")
    
    print(f"\nIntensity Statistics")
    print(f"Mean: {np.mean(y_int_train):.3f}")
    print(f"Std: {np.std(y_int_train):.3f}")
    print(f"Min: {np.min(y_int_train):.3f}")
    print(f"Max: {np.max(y_int_train):.3f}")
    
    print(f"\nSpeed Statistics (m/s)")
    print(f"Mean: {np.mean(X_train[:, -1, 0]):.2f}")
    print(f"Std: {np.std(X_train[:, -1, 0]):.2f}")
    print(f"Min: {np.min(X_train[:, -1, 0]):.2f}")
    print(f"Max: {np.max(X_train[:, -1, 0]):.2f}")
    
    print(f"\nRealistic EV braking dataset saved to {save_path}")
    
    return X, y_class, y_intensity


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate dataset
    X, y_class, y_intensity = generate_realistic_dataset(
        n_samples=15000,
        save_path="modules/braking/data"
    )
    
    print("\nRealistic EV braking simulation completed")
