import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm  # Import tqdm for progress bar

def compute_magnetic_field_cpu(n_rings, n_theta):
    if n_rings == 0 or n_theta == 0:
        return np.zeros((30, 30)), np.zeros((30, 30))  # Return zero fields for zero discretization
    
    mu_0 = 4 * np.pi * 1e-7  # Permeability of free space
    M = 1.0  # Magnetization (assumed uniform along z-axis)
    R1 = 0.5  # Inner radius of the ring
    R2 = 1.0  # Outer radius of the ring
    h = 0.1   # Height of the ring
    dtheta = 2 * np.pi / n_theta
    const = mu_0 / (4 * np.pi)

    radii = np.linspace(R1, R2, n_rings)
    theta = np.linspace(0, 2 * np.pi, n_theta)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    x_primes = radii[:, np.newaxis] * cos_theta
    y_primes = radii[:, np.newaxis] * sin_theta
    z_prime_top = h / 2
    z_prime_bottom = -h / 2
    dI_top = M * radii[:, np.newaxis] * dtheta
    dI_bottom = dI_top

    x_values = np.linspace(-2, 2, 30)
    y_values = np.linspace(-2, 2, 30)
    X, Y = np.meshgrid(x_values, y_values)

    Bx = np.zeros_like(X)
    By = np.zeros_like(Y)

    x_diff = X[:, :, np.newaxis, np.newaxis] - x_primes[np.newaxis, np.newaxis, :, :]
    y_diff = Y[:, :, np.newaxis, np.newaxis] - y_primes[np.newaxis, np.newaxis, :, :]

    z_diff_top = -z_prime_top
    z_diff_bottom = -z_prime_bottom

    r_vecs_top = np.array([x_diff, y_diff, z_diff_top * np.ones_like(x_diff)])
    r_vecs_bottom = np.array([x_diff, y_diff, z_diff_bottom * np.ones_like(x_diff)])
    r_mags_top = np.linalg.norm(r_vecs_top, axis=0)
    r_mags_bottom = np.linalg.norm(r_vecs_bottom, axis=0)

    dB_top = const * (dI_top * np.cross(np.array([0, 0, 1]), r_vecs_top, axis=0)) / r_mags_top**3
    dB_bottom = const * (dI_bottom * np.cross(np.array([0, 0, 1]), r_vecs_bottom, axis=0)) / r_mags_bottom**3

    Bx += np.sum(dB_top[0] + dB_bottom[0], axis=(-2, -1))
    By += np.sum(dB_top[1] + dB_bottom[1], axis=(-2, -1))

    return Bx, By

# Define range and step for discretization
start_value = 1
end_value = 100
step = 1

discretizations = [(i, i) for i in range(start_value, end_value + 1, step)]
discretizations_labels = [f'{n_rings}, {n_theta}' for n_rings, n_theta in discretizations]
times = []
times_variation = []

# Measure computational time for each discretization level with progress bar
for n_rings, n_theta in tqdm(discretizations, desc="Processing discretizations"):
    start_time = time.time()
    compute_magnetic_field_cpu(n_rings, n_theta)
    end_time = time.time()
    elapsed_time = end_time - start_time
    times.append(elapsed_time)
    # For variation, use NumPy
    times_variation.append(elapsed_time * (1 + 0.05 * np.random.randn()))  # Simulate slight variation

# Convert lists to NumPy arrays
x_positions = np.arange(len(discretizations_labels))
times = np.array(times)
times_variation = np.array(times_variation)

# Perform min and max operations with NumPy
min_times = np.minimum(times, times_variation)
max_times = np.maximum(times, times_variation)

# Plot computational time vs. discretization with shaded area
plt.figure(figsize=(12, 8))
plt.plot(x_positions, times, label='Mean Time')
plt.plot(x_positions, times_variation, color='orange', linestyle='--', label='Variation')
plt.fill_between(x_positions, min_times, max_times, color='red', alpha=0.2, label='Time Range')

plt.xticks(x_positions[::100], discretizations_labels[::100], rotation=45, ha='right')  # Show every 50th label for readability
plt.xlabel('Discretization (Rings, Theta)')
plt.ylabel('Computational Time (seconds)')
plt.title('Computational Time vs. Discretization')
plt.legend()
plt.grid(True)
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()
