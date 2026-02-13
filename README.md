# Magnetic Field of Cylindrical Axially-Magnetized Ring
## Project Description

This project simulates and visualizes the magnetic field produced by a cylindrical axially-magnetized ring in the x-y plane. The simulation employs the Biot-Savart law to calculate the magnetic field. It includes both vector field visualization and a field strength heatmap as well as a Field Perturbation Sensitivity Heatmap and MCMC corner plot. Feedback and contributions are welcome to improve its accuracy and functionality. <br> 

![Screenshot_13](https://github.com/user-attachments/assets/f0f17dce-511f-4012-a76f-e97988bcac04)


## Methodology

- The magnetic field is computed using a brute-force approach, iterating over a grid of points and summing the contributions from differential current elements on the ring. This method provides a straightforward but computationally intensive way to calculate the magnetic field.
- The Field Perturbation Sensitivity Heatmap provides a detailed visualization of the field changes for a small spatial perturbation, which could indicate how chaotic or stable the spatial magnetic field is in certain regions.

## Vector Field Visualization
![yee](https://github.com/user-attachments/assets/d46c791b-ff2f-42f9-b385-8f35f7633a50)

## Field Perturbation Sensitivity Heatmap
![heat](https://github.com/user-attachments/assets/042efb1a-76aa-4312-8d36-bcf335230099)

## Computational Time Benchmark 
![d6ea07ff-2aa0-441a-8c27-379f638f6a36](https://github.com/user-attachments/assets/d8d45c51-e8bf-4d83-a2af-955e7d343bee)


## Acknowledgement
Inspired by Luyu-wu project :)))
