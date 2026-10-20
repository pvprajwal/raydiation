# Raydiation
Ray tracing for radiative heat transfer in CUDA. CS6023 Course Project.

The domain is split into (n x n) volume cells. Every cell emits the same number of rays. Each GPU thread traces rays one at a time until they are absorbed in a volume cell or on a wall, and counts the result in a (source cell, destination) table.

Output: raydiation_output.csv. r[s][d] is the fraction of rays from source cell s that were absorbd at destination d.

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