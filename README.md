# Magnetic Field of Cylindrical Magnetic Ring
## Project Description

This project simulates and visualizes the magnetic field produced by a cylindrical magnetic ring in the x-y plane. The simulation employs the Biot-Savart law to calculate the magnetic field and includes both vector field visualization and a field strength heatmap. This is an early beta version of the simulation. As such, the code might contain errors or inaccuracies. Feedback and contributions are welcome to improve its accuracy and functionality. 

```math
\mathbf{dB} = \frac{\mu_0}{4\pi} \frac{\mathbf{dI} \times \mathbf{r}}{|\mathbf{r}|^3}
```

## Methodology

The magnetic field is computed using a brute-force approach, iterating over a grid of points and summing the contributions from differential current elements on the ring. This method provides a straightforward but computationally intensive way to calculate the magnetic field.

## Changable Parameters 
```
# The higher the value, the longer the computational time
n_rings = 100
n_theta = 100
```

## Vector Field Visualization
![Figure_1](https://github.com/user-attachments/assets/5874ded3-4669-4733-9a7f-8561535f91be)
![lol](https://github.com/user-attachments/assets/9b61191b-cee1-4daf-bc1c-e25d0de7d086)


## Computational Time Benchmark
![Figure_2](https://github.com/user-attachments/assets/e1824c56-e181-4dfa-9645-5f4d7fb8345a)

## Discretization Error
![Screenshot_1106](https://github.com/user-attachments/assets/29a51628-150f-4781-bbb0-a24ae7e6c59d)
(This thing run for 2 hours+, will improve it soon)

## Acknowledgement
Inspired by Luyu-wu project :)))
