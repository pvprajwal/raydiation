#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "model.cuh"

// one thread per ray in strided fashion
// tallies = (cells^2) -> ndest map. ndest = cells^2 + 4*cells (volume cells + wall edges)

__global__ void trace_rays_kernel(int total_rays, int rays_per_cell, int seed, int ndest, bool volume_3d, Model model, TraceConfig cfg, int* tallies) {
    int stride = gridDim.x * blockDim.x;

    for (int ray = blockIdx.x * blockDim.x + threadIdx.x; ray<total_rays; ray += stride) {

        Rng rng = make_rng(seed, ray);

        int source = (ray/rays_per_cell);
        int sx = source%cfg.cells, sy = source/cfg.cells;

        // random point inside the source cell.
        float x0=(sx+(float)uniform01(rng))/cfg.cells, y0 =(sy+(float)uniform01(rng))/cfg.cells;

        float path_scale = 1.0; // scale factor for 3D 
        if (volume_3d) {
            float cos_polar = 2.0*(float)uniform01(rng) - 1.0;
            path_scale = sqrtf(fmaxf(1.0f - cos_polar*cos_polar, 0.0));
        }
        float angle = PI*2 * (float)uniform01(rng);

        float n0 = model.n(x0, y0);
        State start{x0, y0, n0*cosf(angle), n0*sinf(angle)};

        // follow the ray, then count where it ended.
        int dest = trace_ray(start, path_scale, model, cfg, rng);
        if (dest < 0) continue; // bad ray

        atomicAdd(&tallies[source*ndest + dest], 1);
    }
}

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error: %s (line %d)\n",                     \
                    cudaGetErrorString(err), __LINE__);                       \
            exit(1);                                                          \
        }                                                                     \
    } while (0)


int main() {

    // PROBLEM PARAMETERS AND ASSUMPTIONS
    
    int N = 20;                       // volume cells per side
    int isGRIN = 1;                   // 1 = graded index, 0 = uniform index
    int rays_per_cell = 4096;
    int seed = 6023;
    float tau = 1.0f;                     // optical thickness
    float omega = 0.0f;                   // scattering albedo
    float max_step = 0.005f;              // longest step, fraction of H
    float wall_emissivity = 1.0f;         // 1 = black walls
    int wall_angle = WallProjected3D;     // for gray walls
    bool volume_3d = false;
    std::string out_dir = "out_gpu";

    int tot= N*N;          // number of volume cells
    int ndest= tot + 4*N;      // cells + wall segments
    int total_rays= tot * rays_per_cell;

    Model model = (isGRIN) ? Model{5.0f, 0.4356f} : Model{1.0f, 0.0f};
    TraceConfig cfg{tau, omega, max_step, N, wall_emissivity, wall_angle};

    size_t tally_count = (size_t)tot * ndest;
    int* d_tallies;
    CUDA_CHECK(cudaMalloc(&d_tallies, tally_count *sizeof(int)));
    CUDA_CHECK(cudaMemset(d_tallies, 0, tally_count *sizeof(int)));

    int block = 256;
    int grid = std::min((total_rays+block-1)/block, 4096);

    cudaEvent_t start_ev, stop_ev;
    CUDA_CHECK(cudaEventCreate(&start_ev));
    CUDA_CHECK(cudaEventCreate(&stop_ev));

    CUDA_CHECK(cudaEventRecord(start_ev));
    trace_rays_kernel<<<grid, block>>>(total_rays, rays_per_cell, seed, ndest, volume_3d, model, cfg, d_tallies);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_ev));
    CUDA_CHECK(cudaEventSynchronize(stop_ev));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_ev, stop_ev));
    printf("Traced %d rays in %.2f ms\n", total_rays, ms);

    std::vector<int> tallies(tally_count);
    CUDA_CHECK(cudaMemcpy(tallies.data(), d_tallies, tally_count*sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_tallies);

    // csv output
    auto label = [&](int id) {
        if (id < tot) return "cell_y" + std::to_string(id/N) + "_x" + std::to_string(id%N);
        int wall = id-tot;
        std::string seg = std::to_string(wall%N);
        const char* names[] = {"wall_left_y", "wall_right_y", "wall_bottom_x", "wall_top_x"};
        return names[wall / N] + seg;
    };

    std::filesystem::create_directories(out_dir);
    std::ofstream csv(out_dir + "/raydiation_output.csv");

    csv << "source_label";
    for (int d = 0; d < ndest; ++d) csv << ',' << label(d);
    csv << '\n' << std::setprecision(10);

    for (int s = 0; s < tot; ++s) {
        csv << label(s);
        for (int d = 0; d < ndest; ++d)
            csv << ',' << (double)tallies[(size_t)s * ndest + d] / rays_per_cell;
        csv << '\n';
    }

    std::cout << "Wrote " << out_dir << "/raydiation_output.csv\n";
    return 0;
}
