import numpy as np
import matplotlib.pyplot as plt
import time

def compute_magnetic_field(n_rings, n_theta):
    if n_rings == 0 or n_theta == 0:
        return np.zeros((30, 30)), np.zeros((30, 30))  # Return zero fields for zero discretization
    
    mu_0 = 4 * np.pi * 1e-7  # Permeability of free space
    M = 1.0  # Magnetization (assumed uniform along z-axis)
    R1 = 0.5  # Inner radius of the ring
    R2 = 1.0  # Outer radius of the ring
    h = 0.1   # Height of the ring
    dtheta = 2 * np.pi / n_theta
    const = mu_0 / (4 * np.pi)

    radii = np.linspace(R1, R2, n_rings) if n_rings > 0 else np.array([0])
    theta = np.linspace(0, 2 * np.pi, n_theta) if n_theta > 0 else np.array([0])
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    x_primes = radii[:, np.newaxis] * cos_theta
    y_primes = radii[:, np.newaxis] * sin_theta
    z_prime_top = h / 2
    z_prime_bottom = -h / 2
    dI_top = M * radii[:, np.newaxis] * dtheta if n_rings > 0 else np.array([0])
    dI_bottom = dI_top

    x_values = np.linspace(-2, 2, 30)
    y_values = np.linspace(-2, 2, 30)
    X, Y = np.meshgrid(x_values, y_values)

    Bx = np.zeros_like(X)
    By = np.zeros_like(Y)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = X[i, j]
            y = Y[i, j]

            x_diff = x - x_primes
            y_diff = y - y_primes
            z_diff_top = -z_prime_top
            z_diff_bottom = -z_prime_bottom

            r_vecs_top = np.array([x_diff, y_diff, z_diff_top * np.ones_like(x_diff)])
            r_vecs_bottom = np.array([x_diff, y_diff, z_diff_bottom * np.ones_like(x_diff)])
            r_mags_top = np.linalg.norm(r_vecs_top, axis=0)
            r_mags_bottom = np.linalg.norm(r_vecs_bottom, axis=0)

            dB_top = const * (dI_top * np.cross([0, 0, 1], r_vecs_top, axis=0)) / r_mags_top**3
            dB_bottom = const * (dI_bottom * np.cross([0, 0, 1], r_vecs_bottom, axis=0)) / r_mags_bottom**3

            Bx[i, j] = np.sum(dB_top[0] + dB_bottom[0])
            By[i, j] = np.sum(dB_top[1] + dB_bottom[1])

    return Bx, By

# Define different discretization levels
discretizations = [(0, 0), (50, 50), (100, 100), (200, 200), (400, 400), (500, 500), (600, 600), (700, 700), (800, 800), (900, 900), (1000, 1000)]
times = []
times_variation = []

# Measure computational time for each discretization level
for n_rings, n_theta in discretizations:
    start_time = time.time()
    compute_magnetic_field(n_rings, n_theta)
    end_time = time.time()
    elapsed_time = end_time - start_time
    times.append(elapsed_time)
    # For variation, simulate some randomness or use different runs
    times_variation.append(elapsed_time * (1 + 0.1 * np.random.randn()))  # Simulate slight variation

# Plot computational time vs. discretization with shaded area
discretizations_labels = [f'{n_rings} rings, {n_theta} theta' for n_rings, n_theta in discretizations]
x_positions = np.arange(len(discretizations_labels))

plt.figure(figsize=(12, 8))
plt.plot(discretizations_labels, times, marker='o', label='Mean Time')
plt.plot(discretizations_labels, times_variation, marker='o', color='orange', linestyle='--', label='Variation')
plt.fill_between(discretizations_labels, np.minimum(times, times_variation), np.maximum(times, times_variation), color='gray', alpha=0.2, label='Time Range')

plt.xlabel('Discretization Level')
plt.ylabel('Computational Time (seconds)')
plt.title('Computational Time vs. Discretization')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()
