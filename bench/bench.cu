#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "model.cuh"
#include "adaptive.cuh"

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error: %s (line %d)\n",                     \
                    cudaGetErrorString(err), __LINE__);                       \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// v1 fixed-step trace.
__global__ void trace_fixed_kernel(int total_rays, int rays_per_cell, int seed, int ndest, bool volume_3d, Model model, TraceConfig cfg, int* tallies) {
    int stride = gridDim.x * blockDim.x;

    for (int ray = blockIdx.x * blockDim.x + threadIdx.x; ray < total_rays; ray += stride) {
        Rng rng = make_rng(seed, ray);

        int source = (ray/rays_per_cell);
        int sx = source%cfg.cells, sy = source/cfg.cells;

        float x0=(sx+(float)uniform01(rng))/cfg.cells, y0 =(sy+(float)uniform01(rng))/cfg.cells;

        float path_scale = 1.0;
        if (volume_3d) {
            float cos_polar = 2.0*(float)uniform01(rng) - 1.0;
            path_scale = sqrtf(fmaxf(1.0f - cos_polar*cos_polar, 0.0));
        }
        float angle = PI*2 * (float)uniform01(rng);

        float n0 = model.n(x0, y0);
        State start{x0, y0, n0*cosf(angle), n0*sinf(angle)};

        int dest = trace_ray(start, path_scale, model, cfg, rng);
        if (dest < 0) continue;

        atomicAdd(&tallies[source*ndest + dest], 1);
    }
}

// adaptive trace
__global__ void __launch_bounds__(256, 4) trace_adaptive_kernel(int total_rays, int rays_per_cell, int seed, int ndest, bool volume_3d, Model model, TraceConfig cfg, AdaptiveConfig ac, int* tallies) {
    int stride = gridDim.x * blockDim.x;

    for (int ray = blockIdx.x * blockDim.x + threadIdx.x; ray < total_rays; ray += stride) {
        Rng rng = make_rng(seed, ray);

        int source = (ray/rays_per_cell);
        int sx = source%cfg.cells, sy = source/cfg.cells;

        float x0=(sx+(float)uniform01(rng))/cfg.cells, y0 =(sy+(float)uniform01(rng))/cfg.cells;

        float path_scale = 1.0;
        if (volume_3d) {
            float cos_polar = 2.0*(float)uniform01(rng) - 1.0;
            path_scale = sqrtf(fmaxf(1.0f - cos_polar*cos_polar, 0.0));
        }
        float angle = PI*2 * (float)uniform01(rng);

        float n0 = model.n(x0, y0);
        State start{x0, y0, n0*cosf(angle), n0*sinf(angle)};

        int dest = trace_ray_adaptive(start, path_scale, model, cfg, ac, rng);
        if (dest < 0) continue;

        atomicAdd(&tallies[source*ndest + dest], 1);
    }
}

struct Args {
    std::string integrator = "adaptive";
    int cells = 20;
    int rays_per_cell = 4096;
    int seed = 6023;
    int block = 256;
    int warmup = 2;
    int repeats = 10;
    float tol = 1e-6f;
    bool verify = false;
};

static void usage(const char* argv0) {
    printf("Usage: %s [options]\n", argv0);
    printf("  --integrator fixed|adaptive  (default adaptive)\n");
    printf("  --cells N                    cells per side (default 20)\n");
    printf("  --rays-per-cell N            rays per source cell (default 4096)\n");
    printf("  --seed N                     RNG seed (default 6023)\n");
    printf("  --block N                    launch block size (default 256)\n");
    printf("  --warmup N                   warmup launches (default 2)\n");
    printf("  --repeats N                  timed launches (default 10)\n");
    printf("  --tol X                      adaptive RK4 tolerance (default 1e-6)\n");
    printf("  --verify                     check final row sums (small grids)\n");
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { usage(argv[0]); exit(0); }
        if (arg == "--verify") { a.verify = true; continue; }
        if (i + 1 >= argc) { fprintf(stderr, "missing value after %s\n", arg.c_str()); exit(1); }
        if (arg == "--integrator") a.integrator = argv[++i];
        else if (arg == "--cells") a.cells = atoi(argv[++i]);
        else if (arg == "--rays-per-cell") a.rays_per_cell = atoi(argv[++i]);
        else if (arg == "--seed") a.seed = atoi(argv[++i]);
        else if (arg == "--block") a.block = atoi(argv[++i]);
        else if (arg == "--warmup") a.warmup = atoi(argv[++i]);
        else if (arg == "--repeats") a.repeats = atoi(argv[++i]);
        else if (arg == "--tol") a.tol = (float)atof(argv[++i]);
        else { fprintf(stderr, "unknown option: %s\n", arg.c_str()); exit(1); }
    }
    if (a.integrator != "fixed" && a.integrator != "adaptive") {
        fprintf(stderr, "--integrator must be fixed or adaptive\n"); exit(1);
    }
    if (a.cells < 1 || a.rays_per_cell < 1 || a.block < 32 || a.block > 256 ||
        a.warmup < 0 || a.repeats < 1 || a.tol <= 0.0f) {
        fprintf(stderr, "invalid grid, ray count, block, repeat, or tolerance\n"); exit(1);
    }
    return a;
}

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);
    int nc = a.cells*a.cells;
    int ndest = nc + 4*a.cells;
    long long total_rays = (long long)nc*a.rays_per_cell;
    if (total_rays > INT_MAX) {
        fprintf(stderr, "total rays exceed int range\n"); exit(1);
    }

    Model model{5.0f, 0.4356f};
    TraceConfig cfg{1.0f, 0.0f, 0.005f, a.cells, 1.0f, WallProjected3D};
    AdaptiveConfig ac;
    ac.tol = a.tol;
    ac.boundary_step = cfg.max_step;

    size_t tally_count = (size_t)nc*ndest;
    int* d_tallies;
    CUDA_CHECK(cudaMalloc(&d_tallies, tally_count*sizeof(int)));

    int block = a.block;
    int grid = std::min((int)((total_rays + block - 1)/block), 4096);

    cudaEvent_t start_ev, stop_ev;
    CUDA_CHECK(cudaEventCreate(&start_ev));
    CUDA_CHECK(cudaEventCreate(&stop_ev));

    auto launch = [&]() {
        if (a.integrator == "fixed") {
            trace_fixed_kernel<<<grid, block>>>((int)total_rays, a.rays_per_cell, a.seed, ndest, false, model, cfg, d_tallies);
        } else {
            trace_adaptive_kernel<<<grid, block>>>((int)total_rays, a.rays_per_cell, a.seed, ndest, false, model, cfg, ac, d_tallies);
        }
        CUDA_CHECK(cudaGetLastError());
    };

    for (int w = 0; w < a.warmup; ++w) {
        CUDA_CHECK(cudaMemset(d_tallies, 0, tally_count*sizeof(int)));
        launch();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> times;
    times.reserve(a.repeats);
    for (int r = 0; r < a.repeats; ++r) {
        CUDA_CHECK(cudaMemset(d_tallies, 0, tally_count*sizeof(int)));
        CUDA_CHECK(cudaEventRecord(start_ev));
        launch();
        CUDA_CHECK(cudaEventRecord(stop_ev));
        CUDA_CHECK(cudaEventSynchronize(stop_ev));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_ev, stop_ev));
        times.push_back(ms);
    }

    std::sort(times.begin(), times.end());
    float median = times[a.repeats/2];
    float min_ms = times.front(), max_ms = times.back();
    double rays_per_s = (double)total_rays/(median*1e-3);

    printf("%s,%d,%d,%lld,%d,%.6g,%d,%d,%.4f,%.4f,%.4f,%.3f\n",
           a.integrator.c_str(), a.cells, a.rays_per_cell, total_rays, block,
           a.tol, a.warmup, a.repeats, median, min_ms, max_ms, rays_per_s);

    if (a.verify) {
        std::vector<int> tallies(tally_count);
        CUDA_CHECK(cudaMemcpy(tallies.data(), d_tallies, tally_count*sizeof(int), cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int i = 0; i < nc; ++i) {
            long long sum = 0;
            for (int j = 0; j < ndest; ++j) sum += tallies[(size_t)i*ndest + j];
            if (sum != a.rays_per_cell) bad++;
        }
        fprintf(stderr, "VERIFY rows=%d bad=%d\n", nc, bad);
    }

    cudaFree(d_tallies);
    cudaEventDestroy(start_ev);
    cudaEventDestroy(stop_ev);
    return 0;
}
