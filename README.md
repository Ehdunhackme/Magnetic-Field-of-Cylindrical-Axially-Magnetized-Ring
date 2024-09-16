# Magnetic Field of Cylindrical Magnetic Ring
## Project Description

This project simulates and visualizes the magnetic field produced by a cylindrical magnetic ring in the x-y plane. The simulation employs the Biot-Savart law to calculate the magnetic field and includes both vector field visualization and a field strength heatmap as well as a Lyapunov Exponent heatmap and MCMC corner plot. This is an early beta version of the simulation. As such, the code might contain errors or inaccuracies. Feedback and contributions are welcome to improve its accuracy and functionality. 

## Methodology

- The magnetic field is computed using a brute-force approach, iterating over a grid of points and summing the contributions from differential current elements on the ring. This method provides a straightforward but computationally intensive way to calculate the magnetic field.
- The Lyapunov heatmap provides a detailed visualization of the system's sensitivity to small perturbations by calculating the Lyapunov exponent at each point in the magnetic field.
- Markov chain Monte Carlo Plot is used to estimate the parameters of a physical system involving magnetic fields and Lyapunov exponents, fitting a model to observational data to infer the best-fit values and their uncertainties.

## Vector Field Visualization
![magnet](https://github.com/user-attachments/assets/3ef61d74-269f-4232-b30e-ed943e8ed94e)

## Lyapunov Exponent Heatmap
![heatmap](https://github.com/user-attachments/assets/5e1de61e-bd20-4f3f-b580-8218b7a10453)

## MCMC Corner Plot
![mcmc dd](https://github.com/user-attachments/assets/0a39224a-2753-4854-b1b2-8d54c0fca85d)
- The parameters are well-constrained with tight uncertainties
- There is some degree of correlation between radii parameters (R1 and R2), but other pairs show weaker correlations

## Computational Time Benchmark 
![d6ea07ff-2aa0-441a-8c27-379f638f6a36](https://github.com/user-attachments/assets/d8d45c51-e8bf-4d83-a2af-955e7d343bee)


## Acknowledgement
Inspired by Luyu-wu project :)))
