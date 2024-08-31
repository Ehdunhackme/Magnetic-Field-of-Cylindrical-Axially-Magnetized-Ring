import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def compute_magnetic_field(n_rings, n_theta):
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

def calculate_error(reference_Bx, reference_By, Bx, By):
    error = np.sqrt((reference_Bx - Bx)**2 + (reference_By - By)**2)
    return np.max(error)

# Define different discretization levels
n_rings_list = [50, 100, 200, 400, 500, 600, 700, 800, 900, 1000]
n_theta_list = [50, 100, 200, 400, 500, 600, 700, 800, 900, 1000]
errors = np.zeros((len(n_rings_list), len(n_theta_list)))

# Compute reference high-resolution field
reference_Bx, reference_By = compute_magnetic_field(1000, 1000)  # Use a high resolution for the reference

# Calculate errors for different discretization levels
for i, n_rings in enumerate(n_rings_list):
    for j, n_theta in enumerate(n_theta_list):
        if n_rings > 0 and n_theta > 0:
            Bx, By = compute_magnetic_field(n_rings, n_theta)
            error = calculate_error(reference_Bx, reference_By, Bx, By)
            errors[i, j] = error
        else:
            errors[i, j] = np.nan  # Set to NaN if computation is not possible

# Plot the error vs. discretization level in 3D
X, Y = np.meshgrid(n_theta_list, n_rings_list)
Z = errors

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

ax.set_xlabel('Theta Discretization')
ax.set_ylabel('Ring Discretization')
ax.set_zlabel('Maximum Error')
ax.set_title('Discretization Error vs. Discretization Levels')
plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
plt.show()
