# Magnetic Field of Cylindrical Magnetic Ring
## Project Description

This project simulates and visualizes the magnetic field produced by a cylindrical magnetic ring in the x-y plane. The simulation employs the Biot-Savart law to calculate the magnetic field and includes both vector field visualization and a field strength heatmap. This is an early beta version of the simulation. As such, the code might contain errors or inaccuracies. Feedback and contributions are welcome to improve its accuracy and functionality. 

## Methodology

The magnetic field is computed using a brute-force approach, iterating over a grid of points and summing the contributions from differential current elements on the ring. This method provides a straightforward but computationally intensive way to calculate the magnetic field.

## Changable Parameters 
```
# The higher the value, the longer the computational time
M = 1.0  # Magnetization (assumed uniform along the z-axis)
R1 = 0.5  # Inner radius of the ring
R2 = 1  # Outer radius of the ring
h = 0.1   # Height of the ring
n_rings = 100
n_theta = 100
```

## Vector Field Visualization
![Figure_dawdad1](https://github.com/user-attachments/assets/b22f4abf-efed-4924-add3-0e6b237a86a4)

## Computational Time Benchmark 
![d6ea07ff-2aa0-441a-8c27-379f638f6a36](https://github.com/user-attachments/assets/d8d45c51-e8bf-4d83-a2af-955e7d343bee)



## Acknowledgement
Inspired by Luyu-wu project :)))
