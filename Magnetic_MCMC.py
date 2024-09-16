import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from tqdm import tqdm

# Define constants and parameters
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space
n_rings = 100
n_theta = 100
dtheta = 2 * np.pi / n_theta  # Small increment of the angle
const = mu_0 / (4 * np.pi)
perturbation = 1e-3  # Small perturbation for Lyapunov exponent calculation

def calculate_magnetic_field(X, Y, Bx, By, B_mag, x_primes, y_primes, z_diff_top, z_diff_bottom, dI_top, dI_bottom, const):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = X[i, j]
            y = Y[i, j]
            x_diff = x - x_primes
            y_diff = y - y_primes
            r_mags_top = np.sqrt(x_diff**2 + y_diff**2 + z_diff_top**2)
            r_mags_bottom = np.sqrt(x_diff**2 + y_diff**2 + z_diff_bottom**2)
            dB_top_x = const * (dI_top * (y_diff)) / r_mags_top**3
            dB_top_y = const * (-dI_top * (x_diff)) / r_mags_top**3
            dB_bottom_x = const * (dI_bottom * (y_diff)) / r_mags_bottom**3
            dB_bottom_y = const * (-dI_bottom * (x_diff)) / r_mags_bottom**3
            Bx[i, j] = np.sum(dB_top_x + dB_bottom_x)
            By[i, j] = np.sum(dB_top_y + dB_bottom_y)
            B_mag[i, j] = np.sqrt(Bx[i, j]**2 + By[i, j]**2)

def calculate_lyapunov_exponent(X, Y, lyapunov_exponent, Bx, By, perturbation, const, x_primes, y_primes, z_diff_top, z_diff_bottom, dI_top, dI_bottom):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = X[i, j]
            y = Y[i, j]
            Bx_orig = Bx[i, j]
            By_orig = By[i, j]
            x_perturbed = x + perturbation
            y_perturbed = y + perturbation
            x_diff_perturbed = x_perturbed - x_primes
            y_diff_perturbed = y_perturbed - y_primes
            r_mags_top_perturbed = np.sqrt(x_diff_perturbed**2 + y_diff_perturbed**2 + z_diff_top**2)
            r_mags_bottom_perturbed = np.sqrt(x_diff_perturbed**2 + y_diff_perturbed**2 + z_diff_bottom**2)
            dB_top_x_perturbed = const * (dI_top * (y_diff_perturbed)) / r_mags_top_perturbed**3
            dB_top_y_perturbed = const * (-dI_top * (x_diff_perturbed)) / r_mags_top_perturbed**3
            dB_bottom_x_perturbed = const * (dI_bottom * (y_diff_perturbed)) / r_mags_bottom_perturbed**3
            dB_bottom_y_perturbed = const * (-dI_bottom * (x_diff_perturbed)) / r_mags_bottom_perturbed**3
            Bx_perturbed = np.sum(dB_top_x_perturbed + dB_bottom_x_perturbed)
            By_perturbed = np.sum(dB_top_y_perturbed + dB_bottom_y_perturbed)
            dBx = Bx_perturbed - Bx_orig
            dBy = By_perturbed - By_orig
            lyapunov_exponent[i, j] = np.sqrt(dBx**2 + dBy**2) / perturbation

def log_likelihood(params):
    M, R1, R2, h = params
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
    x_values = np.linspace(-5, 5, 100)
    y_values = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_values, y_values)
    z_diff_top = -z_prime_top
    z_diff_bottom = -z_prime_bottom
    Bx = np.zeros_like(X)
    By = np.zeros_like(Y)
    B_mag = np.zeros_like(X)
    lyapunov_exponent = np.zeros_like(X)
    calculate_magnetic_field(X, Y, Bx, By, B_mag, x_primes, y_primes, z_diff_top, z_diff_bottom, dI_top, dI_bottom, const)
    calculate_lyapunov_exponent(X, Y, lyapunov_exponent, Bx, By, perturbation, const, x_primes, y_primes, z_diff_top, z_diff_bottom, dI_top, dI_bottom)
    avg_lyapunov = np.mean(lyapunov_exponent)
    
    # Example observational data from the circular current loop
    observed_data = 2e-6  # Magnetic field strength in Tesla

    # Estimate uncertainty as the standard deviation of the parameter samples
    sigma_M = 9.933e-03
    sigma_R1 = 9.710e-03
    sigma_R2 = 9.994e-03
    sigma_h = 1.013e-02

    # Combine uncertainties in a way that's appropriate for your specific model
    # For simplicity, assume these uncertainties are independent and combine them
    sigma_combined = np.sqrt(sigma_M**2 + sigma_R1**2 + sigma_R2**2 + sigma_h**2)

    return -0.5 * ((avg_lyapunov - observed_data) / sigma_combined) ** 2

def log_prior(params):
    M, R1, R2, h = params
    if 0 < M < 20 and 0 < R1 < R2 < 2 and 0 < h < 200:
        return 0.0
    return -np.inf

def log_posterior(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params)

# Set up the MCMC sampler
ndim = 4
nwalkers = 500
nsteps = 20000

# Initialize walkers
initial = np.array([5.0, 0.8, 0.6, 100])
p0 = [initial + 1e-2 * np.random.randn(ndim) for _ in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)

# Run the MCMC sampling with tqdm for progress tracking
print("Running MCMC...")
for _ in tqdm(range(nsteps), desc='MCMC Sampling'):
    sampler.run_mcmc(p0, 1, progress=False)  # Run one step at a time
    p0 = sampler.get_chain()[-1]  # Update the starting positions

# Flatten the chain and convert to DataFrame
samples = sampler.get_chain(discard=100, thin=15, flat=True)

# Plot the corner plot
import corner
fig = corner.corner(samples, labels=["Magnetization (M)", "Inner Radius (R1)", "Outer Radius (R2)", "Height (h)"], 
                     show_titles=True, title_fmt=".2f")
plt.show()

# Compute parameter uncertainties
parameter_names = ["Magnetization (M)", "Inner Radius (R1)", "Outer Radius (R2)", "Height (h)"]
parameter_uncertainties = np.std(samples, axis=0)

for name, uncertainty in zip(parameter_names, parameter_uncertainties):
    print(f"Uncertainty in {name}: {uncertainty:.3e}")

# Estimate uncertainty for the magnetic field model
model_predictions = np.mean(samples, axis=0)
model_std = np.std(samples, axis=0)

# Print the estimated uncertainties for each parameter
print("\nEstimated Uncertainty for Model Parameters:")
for i, param in enumerate(parameter_names):
    print(f"{param}: {model_std[i]:.3e}")
