import numpy as np

# Dataset config
SAMPLING_RATE = 50
WINDOW_SECONDS = 1.5
WINDOW_SIZE = int(SAMPLING_RATE * WINDOW_SECONDS)  # 75 timesteps per window

NUM_SAMPLES_PER_CLASS = 5000
NOISE_STD_SPEED = 0.2
NOISE_STD_BRAKE = 0.03

np.random.seed(42)


def add_noise(signal, std):
    return signal + np.random.normal(0, std, size=signal.shape)


def generate_braking_event(brake_type):

    t = np.linspace(0, WINDOW_SECONDS, WINDOW_SIZE)
    v0 = np.random.uniform(15, 25)

    if brake_type == "light":
        decel = np.random.uniform(0.5, 1.5)
        brake_level = np.random.uniform(0.05, 0.25)

    elif brake_type == "normal":
        decel = np.random.uniform(1.5, 3.5)
        brake_level = np.random.uniform(0.25, 0.6)

    elif brake_type == "emergency":
        decel = np.random.uniform(3.5, 6.0)
        brake_level = np.random.uniform(0.7, 1.0)

    else:
        raise ValueError("Unknown brake type")

    speed = v0 - decel * t
    speed = np.clip(speed, 0, None)

    acceleration = -np.ones_like(t) * decel

    # Smooth ramp-up of brake pedal pressure
    brake_pedal = brake_level * (1 - np.exp(-5 * t))

    speed = add_noise(speed, NOISE_STD_SPEED)
    brake_pedal = add_noise(brake_pedal, NOISE_STD_BRAKE)

    # Final shape: (75, 3)
    sample = np.stack([speed, acceleration, brake_pedal], axis=1)

    return sample


def generate_dataset():
    X = []
    y = []

    classes = {
        "light": 0,
        "normal": 1,
        "emergency": 2
    }

    for brake_type, label in classes.items():
        print(f"Generating {brake_type} braking samples...")

        for _ in range(NUM_SAMPLES_PER_CLASS):
            sample = generate_braking_event(brake_type)
            X.append(sample)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    return X, y


def split_dataset(X, y, train_ratio=0.7, val_ratio=0.15):
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    n_train = int(train_ratio * len(X))
    n_val = int(val_ratio * len(X))

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_val = X[n_train : n_train + n_val]
    y_val = y[n_train : n_train + n_val]

    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":

    print("Generating dataset...")
    X, y = generate_dataset()

    print("Splitting dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)

    print("Saving files...")
    np.save("data/X_train.npy", X_train)
    np.save("data/y_train.npy", y_train)
    np.save("data/X_val.npy", X_val)
    np.save("data/y_val.npy", y_val)
    np.save("data/X_test.npy", X_test)
    np.save("data/y_test.npy", y_test)

    print("Shapes:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)