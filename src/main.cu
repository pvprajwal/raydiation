#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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

// v2 production kernel: prefix[i] is the first global ray index of source
// cell i. A binary search maps each production ray to its source cell, so the
// per-source ray counts can be non-uniform without storing a source array.
__global__ void trace_rays_adaptive_kernel(int total_rays, int seed, const int* prefix, int nc, int ndest, bool volume_3d, Model model, TraceConfig cfg, int* tallies) {
    int stride = gridDim.x * blockDim.x;

    for (int ray = blockIdx.x * blockDim.x + threadIdx.x; ray < total_rays; ray += stride) {

        int lo = 0, hi = nc;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (prefix[mid + 1] <= ray) lo = mid + 1;
            else hi = mid;
        }
        int source = lo;

        Rng rng = make_rng(seed, ray);

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

// one Jacobi sweep for u = T^4 / T_hot^4.
// u = 1 on the hot wall, 0 on the cold walls. D has row sums < 1 because
// the rest of each row is absorbed by the walls, so this converges quickly.
__global__ void temperature_kernel(int nc, int ndest, int n, int hot_wall, int rays_per_cell, const int* rays_per_source, const int* tallies, const float* u, float* un) {
    int stride = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nc; i += stride) {
        const int* row = tallies + i*ndest;
        float sum = 0.0f;

        for (int j = 0; j < nc; ++j) sum += row[j]*u[j];
        for (int seg = 0; seg < n; ++seg) sum += row[nc + hot_wall*n + seg];

        float denom = rays_per_source ? (float)rays_per_source[i] : (float)rays_per_cell;
        un[i] = sum/denom;
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
    int rays_per_cell = 4096;         // average rays per cell (pilot + production)
    int seed = 6023;
    float tau = 1.0f;                     // optical thickness
    float omega = 0.0f;                   // scattering albedo
    float max_step = 0.005f;              // longest step, fraction of H
    float wall_emissivity = 1.0f;         // 1 = black walls
    int wall_angle = WallProjected3D;     // for gray walls
    bool volume_3d = false;
    std::string out_dir = "results";
    float T_hot = 1000.0f;                // hot wall temperature, K
    int hot_wall = 2;                     // 0 left, 1 right, 2 bottom, 3 top

    // v2: pilot batch estimates the per-cell RHS variance; production then
    // allocates n_i ~ sigma_i. Set adaptive = false for v1 uniform rays.
    bool adaptive = true;
    int pilot_rays_per_cell = 512;
    int min_rays_per_cell = 64;

    int tot= N*N;          // number of volume cells
    int ndest= tot + 4*N;      // cells + wall segments
    int total_rays= tot * rays_per_cell;
    int pilot_total= tot * pilot_rays_per_cell;
    int production_total= total_rays - pilot_total;

    if (adaptive && (pilot_rays_per_cell < 1 || pilot_rays_per_cell >= rays_per_cell)) {
        fprintf(stderr, "pilot rays per cell must be in [1, rays per cell)\n");
        return 1;
    }

    Model model = (isGRIN) ? Model{5.0f, 0.4356f} : Model{1.0f, 0.0f};
    TraceConfig cfg{tau, omega, max_step, N, wall_emissivity, wall_angle};

    std::vector<int> rays_per_source(tot, rays_per_cell);
    std::vector<int> prefix;
    std::vector<double> sigma;

    size_t tally_count = (size_t)tot * ndest;
    int* d_tallies;
    CUDA_CHECK(cudaMalloc(&d_tallies, tally_count *sizeof(int)));
    CUDA_CHECK(cudaMemset(d_tallies, 0, tally_count *sizeof(int)));

    int block = 256;
    int grid = std::min((total_rays+block-1)/block, 4096);

    cudaEvent_t start_ev, stop_ev;
    CUDA_CHECK(cudaEventCreate(&start_ev));
    CUDA_CHECK(cudaEventCreate(&stop_ev));

    // u = T^4/T_hot^4 buffers, used by the pilot and final solves
    float *d_u, *d_un;
    CUDA_CHECK(cudaMalloc(&d_u, tot*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_un, tot*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_u, 0, tot*sizeof(float)));

    int tblock = 256;
    int tgrid = std::min((tot+tblock-1)/tblock, 64);
    int titer = 100;

    std::vector<int> tallies(tally_count);

    if (adaptive) {
        // 1. pilot: uniform n0 rays per source cell, independent seed
        int* d_pilot;
        CUDA_CHECK(cudaMalloc(&d_pilot, tally_count *sizeof(int)));
        CUDA_CHECK(cudaMemset(d_pilot, 0, tally_count *sizeof(int)));

        int pilot_grid = std::min((pilot_total+block-1)/block, 4096);
        CUDA_CHECK(cudaEventRecord(start_ev));
        trace_rays_kernel<<<pilot_grid, block>>>(pilot_total, pilot_rays_per_cell, seed, ndest, volume_3d, model, cfg, d_pilot);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop_ev));
        CUDA_CHECK(cudaEventSynchronize(stop_ev));

        float pilot_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&pilot_ms, start_ev, stop_ev));
        printf("Pilot: %d rays in %.2f ms\n", pilot_total, pilot_ms);

        std::vector<int> pilot(tally_count);
        CUDA_CHECK(cudaMemcpy(pilot.data(), d_pilot, tally_count*sizeof(int), cudaMemcpyDeviceToHost));

        // pilot temperature field for the rhs standard-error weights
        for (int it = 0; it < titer; ++it) {
            temperature_kernel<<<tgrid, tblock>>>(tot, ndest, N, hot_wall, pilot_rays_per_cell, nullptr, d_pilot, d_u, d_un);
            CUDA_CHECK(cudaGetLastError());
            float* tmp = d_u; d_u = d_un; d_un = tmp;
        }
        std::vector<float> pilot_u(tot);
        CUDA_CHECK(cudaMemcpy(pilot_u.data(), d_u, tot*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemset(d_u, 0, tot*sizeof(float)));
        cudaFree(d_pilot);

        // 2. per-cell uncertainty in the temperature RHS
        sigma.assign(tot, 0.0);
        for (int i = 0; i < tot; ++i) {
            const int* row = &pilot[(size_t)i*ndest];
            double var = 0.0;
            for (int j = 0; j < ndest; ++j) {
                double p = (double)row[j]/pilot_rays_per_cell;
                double t4 = 0.0;
                if (j < tot) t4 = pilot_u[j];
                else if (j >= tot + 2*N && j < tot + 3*N) t4 = 1.0; // hot bottom wall
                var += p*(1.0-p)/pilot_rays_per_cell * t4*t4;
            }
            sigma[i] = std::sqrt(var);
        }

        double total_sigma = 0.0;
        for (double s : sigma) total_sigma += s;
        if (!(total_sigma > 0.0)) {
            std::fill(sigma.begin(), sigma.end(), 1.0);
            total_sigma = (double)tot;
        }

        // largest-remainder allocation of the production budget
        int min_rays = min_rays_per_cell;
        if ((long long)min_rays*tot > production_total) min_rays = production_total/tot;
        int base = min_rays*tot;
        int remaining = production_total - base;
        std::vector<double> frac(tot, 0.0);
        int assigned = 0;
        for (int i = 0; i < tot; ++i) {
            double ideal = (double)remaining*sigma[i]/total_sigma;
            int fl = (int)std::floor(ideal);
            rays_per_source[i] = min_rays + fl;
            assigned += fl;
            frac[i] = ideal - fl;
        }

        std::vector<int> order(tot);
        for (int i = 0; i < tot; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            if (frac[a] != frac[b]) return frac[a] > frac[b];
            return a < b;
        });
        for (int k = 0; k < remaining-assigned; ++k) rays_per_source[order[k]]++;

        prefix.assign(tot+1, 0);
        for (int i = 0; i < tot; ++i) prefix[i+1] = prefix[i] + rays_per_source[i];

        int lo = rays_per_source[0], hi = rays_per_source[0], sum = 0;
        for (int i = 0; i < tot; ++i) {
            lo = std::min(lo, rays_per_source[i]);
            hi = std::max(hi, rays_per_source[i]);
            sum += rays_per_source[i];
        }
        printf("Allocation: production=%d, n_i min/mean/max=%d/%.1f/%d\n", production_total, lo, (double)sum/tot, hi);

        // 3. production with an independent RNG stream
        int* d_prefix;
        CUDA_CHECK(cudaMalloc(&d_prefix, (tot+1)*sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_prefix, prefix.data(), (tot+1)*sizeof(int), cudaMemcpyHostToDevice));

        int prod_seed = seed ^ 0x5bd1e995;
        int prod_grid = std::min((production_total+block-1)/block, 4096);
        CUDA_CHECK(cudaEventRecord(start_ev));
        trace_rays_adaptive_kernel<<<prod_grid, block>>>(production_total, prod_seed, d_prefix, tot, ndest, volume_3d, model, cfg, d_tallies);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop_ev));
        CUDA_CHECK(cudaEventSynchronize(stop_ev));

        float prod_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&prod_ms, start_ev, stop_ev));
        printf("Production: %d rays in %.2f ms\n", production_total, prod_ms);
        cudaFree(d_prefix);
    } else {
        CUDA_CHECK(cudaEventRecord(start_ev));
        trace_rays_kernel<<<grid, block>>>(total_rays, rays_per_cell, seed, ndest, volume_3d, model, cfg, d_tallies);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop_ev));
        CUDA_CHECK(cudaEventSynchronize(stop_ev));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_ev, stop_ev));
        printf("Traced %d rays in %.2f ms\n", total_rays, ms);
    }

    CUDA_CHECK(cudaMemcpy(tallies.data(), d_tallies, tally_count*sizeof(int), cudaMemcpyDeviceToHost));

    // temperature solve: u = T^4/T_hot^4, fixed 100 Jacobi sweeps
    int* d_n = nullptr;
    if (adaptive) {
        CUDA_CHECK(cudaMalloc(&d_n, tot*sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_n, rays_per_source.data(), tot*sizeof(int), cudaMemcpyHostToDevice));
    }
    for (int it = 0; it < titer; ++it) {
        temperature_kernel<<<tgrid, tblock>>>(tot, ndest, N, hot_wall, rays_per_cell, d_n, d_tallies, d_u, d_un);
        CUDA_CHECK(cudaGetLastError());
        float* tmp = d_u; d_u = d_un; d_un = tmp;
    }

    std::vector<float> u(tot);
    CUDA_CHECK(cudaMemcpy(u.data(), d_u, tot*sizeof(float), cudaMemcpyDeviceToHost));
    if (d_n) cudaFree(d_n);
    cudaFree(d_u);
    cudaFree(d_un);
    cudaFree(d_tallies);
    printf("Solved temperature field in %d Jacobi iterations (hot wall %d, %.0f K)\n", titer, hot_wall, T_hot);

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
            csv << ',' << (double)tallies[(size_t)s * ndest + d] / rays_per_source[s];
        csv << '\n';
    }

    if (adaptive) {
        std::ofstream alloc(out_dir + "/adaptive_allocation.csv");
        alloc << "cell_y,cell_x,cell_index,pilot_rays,production_rays,sigma,prefix_start,prefix_end\n" << std::setprecision(8);
        for (int i = 0; i < tot; ++i)
            alloc << i/N << ',' << i%N << ',' << i << ',' << pilot_rays_per_cell << ',' << rays_per_source[i] << ',' << sigma[i] << ',' << prefix[i] << ',' << prefix[i+1] << '\n';
    }

    std::ofstream temp(out_dir + "/temperature_field.csv");
    temp << "cell_y,cell_x,x_over_H,y_over_H,T_over_Tw1,T_K\n" << std::setprecision(8);
    for (int iy = 0; iy < N; ++iy) {
        for (int ix = 0; ix < N; ++ix) {
            float t = powf(fmaxf(u[iy*N + ix], 0.0f), 0.25f);
            temp << iy << ',' << ix << ',' << (ix+0.5f)/N << ',' << (iy+0.5f)/N << ',' << t << ',' << t*T_hot << '\n';
        }
    }

    std::cout << "Wrote " << out_dir << "/raydiation_output.csv\n";
    std::cout << "Wrote " << out_dir << "/temperature_field.csv\n";
    if (adaptive) std::cout << "Wrote " << out_dir << "/adaptive_allocation.csv\n";
    return 0;
}
