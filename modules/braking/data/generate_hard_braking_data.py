import numpy as np


def generate_hard_sample(seq_len=75):

    speed = np.random.uniform(30, 90)
    accel = 0.0
    brake = 0.0

    X = []

    # Label is based on where braking is heading, not what's happening now
    future_intensity = np.random.rand()

    if future_intensity < 0.35:
        label = 0  # Light
        target_brake = np.random.uniform(0.15, 0.35)
    elif future_intensity < 0.7:
        label = 1  # Normal
        target_brake = np.random.uniform(0.3, 0.6)
    else:
        label = 2  # Emergency
        target_brake = np.random.uniform(0.5, 1.0)

    for t in range(seq_len):
        # Gradually ramp brake toward target
        brake += (target_brake - brake) * np.random.uniform(0.02, 0.08)

        # Driver inconsistency & sensor noise
        brake += np.random.normal(0, 0.05)
        brake = np.clip(brake, 0, 1)

        accel = -brake * np.random.uniform(1.5, 3.5)
        accel += np.random.normal(0, 0.2)

        speed += accel * 0.1
        speed = max(speed, 0)

        X.append([speed, accel, brake])

    return np.array(X), label


def generate_dataset(n_samples=15000):

    X, y = [], []

    for _ in range(n_samples):
        x_i, y_i = generate_hard_sample()
        X.append(x_i)
        y.append(y_i)

    return np.array(X), np.array(y)


if __name__ == "__main__":
    np.random.seed(42)

    X, y = generate_dataset()

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # Split 70/15/15
    n = len(X)
    X_train, y_train = X[:int(0.7 * n)], y[:int(0.7 * n)]
    X_val, y_val = X[int(0.7 * n):int(0.85 * n)], y[int(0.7 * n):int(0.85 * n)]
    X_test, y_test = X[int(0.85 * n):], y[int(0.85 * n):]

    # Save
    np.save("data/X_train_hard.npy", X_train)
    np.save("data/y_train_hard.npy", y_train)
    np.save("data/X_val_hard.npy", X_val)
    np.save("data/y_val_hard.npy", y_val)
    np.save("data/X_test_hard.npy", X_test)
    np.save("data/y_test_hard.npy", y_test)

    print("Hard ambiguous dataset generated.")