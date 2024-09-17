import numpy as np
import matplotlib.pyplot as plt
from numba import jit

# Define constants and parameters
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space
M = 9.0  # Magnetization (assumed uniform along z-axis)
R1 = 1  # Inner radius of the ring
R2 = 0.8  # Outer radius of the ring
h = 100   # Height of the ring
n_rings = 200
n_theta = 500
dtheta = 2 * np.pi / n_theta  # Small increment of the angle
const = mu_0 / (4 * np.pi)
perturbation = 2e-3  # Small perturbation for Lyapunov exponent calculation

# Discretize radii and angles
radii = np.linspace(R1, R2, n_rings)
theta = np.linspace(0, 2 * np.pi, n_theta)
cos_theta = np.cos(theta)
sin_theta = np.sin(theta)

# Precompute positions and current elements
x_primes = radii[:, np.newaxis] * cos_theta
y_primes = radii[:, np.newaxis] * sin_theta
z_prime_top = h / 2
z_prime_bottom = -h / 2
dI_top = M * radii[:, np.newaxis] * dtheta
dI_bottom = dI_top

# Create a grid of points in the x-y plane to visualize the field
x_values = np.linspace(-10, 10, 200)
y_values = np.linspace(-10, 10, 200)
X, Y = np.meshgrid(x_values, y_values)

# Precompute z-component differences (constant)
z_diff_top = -z_prime_top
z_diff_bottom = -z_prime_bottom

Bx = np.zeros_like(X)
By = np.zeros_like(Y)
B_mag = np.zeros_like(X)
lyapunov_exponent = np.zeros_like(X)  # Array to store Lyapunov exponent values

@jit(nopython=True)
def calculate_magnetic_field(X, Y, Bx, By, B_mag, x_primes, y_primes, z_diff_top, z_diff_bottom, dI_top, dI_bottom, const):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = X[i, j]
            y = Y[i, j]
            
            # Calculate differences
            x_diff = x - x_primes
            y_diff = y - y_primes
            
            # Compute the magnitude of r_vecs
            r_mags_top = np.sqrt(x_diff**2 + y_diff**2 + z_diff_top**2)
            r_mags_bottom = np.sqrt(x_diff**2 + y_diff**2 + z_diff_bottom**2)
            
            # Compute the cross product for the Biot-Savart law (2D case)
            dB_top_x = const * (dI_top * (y_diff)) / r_mags_top**3
            dB_top_y = const * (-dI_top * (x_diff)) / r_mags_top**3

            dB_bottom_x = const * (dI_bottom * (y_diff)) / r_mags_bottom**3
            dB_bottom_y = const * (-dI_bottom * (x_diff)) / r_mags_bottom**3
            
            # Sum up the contributions
            Bx[i, j] = np.sum(dB_top_x + dB_bottom_x)
            By[i, j] = np.sum(dB_top_y + dB_bottom_y)
            B_mag[i, j] = np.sqrt(Bx[i, j]**2 + By[i, j]**2)  # Field magnitude

@jit(nopython=True)
def calculate_lyapunov_exponent(X, Y, lyapunov_exponent, Bx, By, perturbation, const, x_primes, y_primes, z_diff_top, z_diff_bottom, dI_top, dI_bottom):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = X[i, j]
            y = Y[i, j]
            
            # Calculate magnetic field at the original point
            Bx_orig = Bx[i, j]
            By_orig = By[i, j]
            
            # Perturb the position slightly
            x_perturbed = x + perturbation
            y_perturbed = y + perturbation
            
            # Recalculate the magnetic field at the perturbed point
            x_diff_perturbed = x_perturbed - x_primes
            y_diff_perturbed = y_perturbed - y_primes
            
            r_mags_top_perturbed = np.sqrt(x_diff_perturbed**2 + y_diff_perturbed**2 + z_diff_top**2)
            r_mags_bottom_perturbed = np.sqrt(x_diff_perturbed**2 + y_diff_perturbed**2 + z_diff_bottom**2)
            
            dB_top_x_perturbed = const * (dI_top * (y_diff_perturbed)) / r_mags_top_perturbed**3
            dB_top_y_perturbed = const * (-dI_top * (x_diff_perturbed)) / r_mags_top_perturbed**3

            dB_bottom_x_perturbed = const * (dI_bottom * (y_diff_perturbed)) / r_mags_bottom_perturbed**3
            dB_bottom_y_perturbed = const * (-dI_bottom * (x_diff_perturbed)) / r_mags_bottom_perturbed**3
            
            # Perturbed magnetic field
            Bx_perturbed = np.sum(dB_top_x_perturbed + dB_bottom_x_perturbed)
            By_perturbed = np.sum(dB_top_y_perturbed + dB_bottom_y_perturbed)
            
            # Calculate the difference between the original and perturbed field
            dBx = Bx_perturbed - Bx_orig
            dBy = By_perturbed - By_orig
            
            # Calculate Lyapunov exponent (rough approximation)
            lyapunov_exponent[i, j] = np.sqrt(dBx**2 + dBy**2) / perturbation

# Run the calculation
calculate_magnetic_field(X, Y, Bx, By, B_mag, x_primes, y_primes, z_diff_top, z_diff_bottom, dI_top, dI_bottom, const)
calculate_lyapunov_exponent(X, Y, lyapunov_exponent, Bx, By, perturbation, const, x_primes, y_primes, z_diff_top, z_diff_bottom, dI_top, dI_bottom)

# Plot the Lyapunov exponent heatmap
plt.figure(figsize=(6, 6))
plt.imshow(lyapunov_exponent, extent=(x_values.min(), x_values.max(), y_values.min(), y_values.max()), origin='lower', aspect='auto', cmap='hot')
plt.colorbar(label='Field Perturbation Sensitivity')
plt.title('Field Perturbation Sensitivity Heatmap')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.show()
