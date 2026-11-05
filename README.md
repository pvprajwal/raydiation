# Raydiation
Raydiation *(Ray tracing + Radiation)* is a CUDA Monte Carlo ray tracing implementation for radiative heat transfer in graded-index media. 

The domain is split into (n x n) volume cells. Every cell emits the same number of rays. Each GPU thread traces rays one at a time until they are absorbed in a volume cell or on a wall, and counts the result in a (source cell, destination) table.

Output: raydiation_output.csv. r[s][d] is the fraction of rays from source cell s that were absorbd at destination d.


The fixed-step advance() in model.cuh integrates with a scalar length
correction. This header uses one full RK4 step and two half steps to
estimate the local error. The two-half-step result is accepted, and the
next step grows or shrinks from that estimate. The last boundary_layer of
distance before a wall is resolved at the v1 step, and collision steps are
corrected onto the sampled free path.

![distribution factors](results/raydiation_visualization.png)   

![temperature fields](results/temperature_profile.png)

Reference:
*Jiang Shao, Keyong Zhu, Yong Huang,
A fast GPU Monte Carlo implementation for radiative heat transfer in graded-index media,
Journal of Quantitative Spectroscopy and Radiative Transfer
https://doi.org/10.1016/j.jqsrt.2021.107680.
(https://www.sciencedirect.com/science/article/pii/S0022407321001734)*


Build:  

`nvcc -O3 -std=c++17 main.cu -o raytrace`

Run:    

`./raytrace `