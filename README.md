# Magnetic Field of Cylindrical Axially Magnetized Ring
## Project Description

This project simulates and visualizes the magnetic field produced by a cylindrical axially magnetized ring in the x-y plane. The simulation employs the Biot-Savart law to calculate the magnetic field. It includes both vector field visualization and a field strength heatmap as well as a Field Perturbation Sensitivity Heatmap and MCMC corner plot. Feedback and contributions are welcome to improve its accuracy and functionality. <br> 

![image](https://github.com/user-attachments/assets/6da84289-c633-4ffb-8587-4645cb9e166b)


## Methodology

- The magnetic field is computed using a brute-force approach, iterating over a grid of points and summing the contributions from differential current elements on the ring. This method provides a straightforward but computationally intensive way to calculate the magnetic field.
- The Field Perturbation Sensitivity Heatmap provides a detailed visualization of the field changes for a small spatial perturbation, which could indicate how chaotic or stable the spatial magnetic field is in certain regions.
- Markov chain Monte Carlo Plot is used to estimate the parameters of a physical system involving magnetic fields and Lyapunov exponents, fitting a model to estimated data to infer the best-fit values and their uncertainties.

## Vector Field Visualization
![yee](https://github.com/user-attachments/assets/d46c791b-ff2f-42f9-b385-8f35f7633a50)

## Field Perturbation Sensitivity Heatmap
![heat](https://github.com/user-attachments/assets/042efb1a-76aa-4312-8d36-bcf335230099)

## MCMC Corner Plot
![mcccccc](https://github.com/user-attachments/assets/8182a861-7b0d-454a-8fad-77b58c1adcb9)
- All the parameter uncertainties were estimated using the standard deviation of the samples for each parameter after running MCMC
- The parameters are well-constrained with tight uncertainties
- There is some correlation between radii parameters (R1 and R2), but other pairs show weaker correlations

## Computational Time Benchmark 
![d6ea07ff-2aa0-441a-8c27-379f638f6a36](https://github.com/user-attachments/assets/d8d45c51-e8bf-4d83-a2af-955e7d343bee)


## Acknowledgement
Inspired by Luyu-wu project :)))
