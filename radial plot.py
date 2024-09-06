import numpy as np
import matplotlib.pyplot as plt
from numba import jit

# Define constants and parameters
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space
M = 9.0  # Magnetization (assumed uniform along z-axis)
R1 = 1  # Inner radius of the ring
R2 = 0.8  # Outer radius of the ring
h = 120# Height of the ring
n_rings = 100
n_theta = 300
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

# Extracting cross-section profiles along x and y axes
x_profile_index = X.shape[0] // 2  # Middle of the x-axis
y_profile_index = Y.shape[1] // 2  # Middle of the y-axis

x_profile = B_mag[x_profile_index, :]
y_profile = B_mag[:, y_profile_index]

# Extract radial profile
def calculate_radial_profile(B_mag, X, Y):
    # Calculate radial distances from the origin (0,0)
    radii = np.sqrt(X**2 + Y**2)
    unique_radii = np.unique(radii)
    radial_profile = np.zeros_like(unique_radii)
    
    for i, r in enumerate(unique_radii):
        mask = (radii >= r - 0.05) & (radii < r + 0.05)  # Adjust tolerance as needed
        radial_profile[i] = np.mean(B_mag[mask])
    
    return unique_radii, radial_profile

radii, radial_profile = calculate_radial_profile(B_mag, X, Y)

# Plotting the profiles
plt.figure(figsize=(14, 6))

# Plot radial profile
plt.plot(radii, radial_profile, label='Radial Profile')
plt.xlabel('Radial Distance (m)')
plt.ylabel('Magnetic Field Strength (T)')
plt.title('Radial Profile')
plt.legend()
plt.show()
