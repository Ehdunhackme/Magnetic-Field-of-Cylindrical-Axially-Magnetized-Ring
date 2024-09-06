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
![Figure_111](https://github.com/user-attachments/assets/887b2b05-987b-4517-bdbe-ca651d5bb9d3)
![Figure_dawdad1](https://github.com/user-attachments/assets/b22f4abf-efed-4924-add3-0e6b237a86a4)

## Acknowledgement
Inspired by Luyu-wu project :)))
