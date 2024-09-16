import numpy as np
import matplotlib.pyplot as plt
from numba import jit

# Define constants and parameters
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space
M = 9.0  # Magnetization (assumed uniform along z-axis)
R1 = 1  # Inner radius of the ring
R2 = 0.8  # Outer radius of the ring
h = 100   # Height of the ring
n_rings = 100
n_theta = 500
dtheta = 2 * np.pi / n_theta  # Small increment of the angle
const = mu_0 / (4 * np.pi)

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
x_values = np.linspace(-2, 2, 30)
y_values = np.linspace(-2, 2, 30)
X, Y = np.meshgrid(x_values, y_values)

# Precompute z-component differences (constant)
z_diff_top = -z_prime_top
z_diff_bottom = -z_prime_bottom

Bx = np.zeros_like(X)
By = np.zeros_like(Y)
B_mag = np.zeros_like(X)

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

# Run the calculation
calculate_magnetic_field(X, Y, Bx, By, B_mag, x_primes, y_primes, z_diff_top, z_diff_bottom, dI_top, dI_bottom, const)

# Plot the magnetic field vectors and field strength
plt.figure(figsize=(12, 10))

# Plot the magnetic field vectors, using B_mag for the color
plt.quiver(X, Y, Bx, By, B_mag, scale=1e-7, pivot='middle', cmap='viridis')
plt.colorbar(label='Magnetic Field Strength (T)')
plt.title('Magnetic Field Vectors Plane')
plt.xlabel('x (m)')
plt.ylabel('y (m)')

# Plot the ring for reference
ring_x = radii[:, np.newaxis] * cos_theta
ring_y = radii[:, np.newaxis] * sin_theta
plt.plot(ring_x, ring_y, 'k-')

plt.show()