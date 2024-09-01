# Magnetic Field of Cylindrical Magnetic Ring
## Project Description

This project simulates and visualizes the magnetic field produced by a cylindrical magnetic ring in the x-y plane. The simulation employs the Biot-Savart law to calculate the magnetic field and includes both vector field visualization and a field strength heatmap. This is an early beta version of the simulation. As such, the code might contain errors or inaccuracies. Feedback and contributions are welcome to improve its accuracy and functionality. 

Biot-Savart law:

```math
\mathbf{dB} = \frac{\mu_0}{4\pi} \frac{\mathbf{dI} \times \mathbf{r}}{|\mathbf{r}|^3}
```

Final Time Complexity (Determined using nested loops): 

```math
\text{Time Complexity} = O\left(n_x \times n_y \times n_{rings} \times n_{theta}\right)
```
<br>

(time complexity grows linearly with the number of grid points and the discretization parameters, planned to optimize it by adding parallel processing or analytical approximation)

## Methodology

The magnetic field is computed using a brute-force approach, iterating over a grid of points and summing the contributions from differential current elements on the ring. This method provides a straightforward but computationally intensive way to calculate the magnetic field.

## Changable Parameters 
```
# The higher the value, the longer the computational time
M = 1.0  # Magnetization (assumed uniform along z-axis)
R1 = 0.5  # Inner radius of the ring
R2 = 1  # Outer radius of the ring
h = 0.1   # Height of the ring
n_rings = 100
n_theta = 100
```

## Vector Field Visualization
![Figure_1](https://github.com/user-attachments/assets/5874ded3-4669-4733-9a7f-8561535f91be)
![Figure_11](https://github.com/user-attachments/assets/24e882d4-2140-4c6c-b3ad-8051b192c953)


## Computational Time Benchmark
![Figure_2](https://github.com/user-attachments/assets/e1824c56-e181-4dfa-9645-5f4d7fb8345a)

## Discretization Error
![Screenshot_1106](https://github.com/user-attachments/assets/29a51628-150f-4781-bbb0-a24ae7e6c59d)
(This thing run for 2 hours+, will improve it soon)

## Acknowledgement
Inspired by Luyu-wu project :)))
