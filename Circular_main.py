import numpy as np
import matplotlib.pyplot as plt

# Define constants and parameters
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space
M = 1.0  # Magnetization (assumed uniform along z-axis)
R1 = 0.5  # Inner radius of the ring
R2 = 1.0  # Outer radius of the ring
h = 0.1   # Height of the ring
n_rings = 100
n_theta = 100
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

Bx = np.zeros_like(X)
By = np.zeros_like(Y)
B_mag = np.zeros_like(X)

# Vectorized calculation of the magnetic field at each grid point
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x = X[i, j]
        y = Y[i, j]
        
        # Broadcast the x and y values to match the shape of x_primes and y_primes
        x_diff = x - x_primes
        y_diff = y - y_primes
        z_diff_top = -z_prime_top
        z_diff_bottom = -z_prime_bottom
        
        # Compute the r_vecs and their magnitudes
        r_vecs_top = np.array([x_diff, y_diff, z_diff_top * np.ones_like(x_diff)])
        r_vecs_bottom = np.array([x_diff, y_diff, z_diff_bottom * np.ones_like(x_diff)])
        r_mags_top = np.linalg.norm(r_vecs_top, axis=0)
        r_mags_bottom = np.linalg.norm(r_vecs_bottom, axis=0)
        
        # Compute the cross product for the Biot-Savart law
        dB_top = const * (dI_top * np.cross([0, 0, 1], r_vecs_top, axis=0)) / r_mags_top**3
        dB_bottom = const * (dI_bottom * np.cross([0, 0, 1], r_vecs_bottom, axis=0)) / r_mags_bottom**3
        
        # Sum up the contributions
        Bx[i, j] = np.sum(dB_top[0] + dB_bottom[0])
        By[i, j] = np.sum(dB_top[1] + dB_bottom[1])
        B_mag[i, j] = np.sqrt(Bx[i, j]**2 + By[i, j]**2)  # Field magnitude

# Plot the magnetic field vectors and field strength
plt.figure(figsize=(12, 6))

# Plot the magnetic field vectors
plt.subplot(1, 2, 1)
plt.quiver(X, Y, Bx, By, scale=1e-7, pivot='middle', color='k')
plt.title('Magnetic Field Vectors (x-y plane)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')

# Plot the field magnitude with a color map
plt.subplot(1, 2, 2)
plt.imshow(B_mag, extent=(x_values.min(), x_values.max(), y_values.min(), y_values.max()), origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='Magnetic Field Strength (T)')
plt.title('Magnetic Field Strength')
plt.xlabel('x (m)')
plt.ylabel('y (m)')

# Plot the ring for reference
ring_x = radii[:, np.newaxis] * cos_theta
ring_y = radii[:, np.newaxis] * sin_theta
plt.subplot(1, 2, 1)
plt.plot(ring_x, ring_y, 'r-' )
plt.legend()

plt.show()
