#pragma once

// Physics for 2D graded-index Monte Carlo ray tracer.
//
// - The domain is the unit square [0,1]^2 
// - Inside, refractive index is a function of position (n = f(x, y)), so rays may be curved
// - The "optical" state is (x, y, tx, ty), where (tx, ty) = dR/dt is the optical direction. 
// - The ray obeys d^2 R / dt^2 = 1/2 grad(n^2)
// - Integrate with 4th-order Runge-Kutta (RK4)

#include <math.h>
#include "rng.cuh"

const float PI = 3.14159265358979323846f;

// How a gray wall re-emits a ray that it reflects.
const int WallCosine2D = 0;    // 2D Lambertian: p(angle) ~ cos(angle)
const int WallIsotropic2D = 1; // every in-plane angle equally likely
const int WallProjected3D = 2; // 3D Lambertian wall, projected to 2D

// Optical State
struct State {
    float x, y, tx, ty;
};

// Refractive index n(x,y) = n_ref * sqrt(1 - a2 * (x^2 + y^2)).
struct Model {
    float n_ref = 1;
    float a2 = 0;       // default is uniform index
    __device__ float n(float x, float y) const {
        return n_ref * sqrtf(fmaxf(1.0f - a2 * (x * x + y * y), 1e-14f));
    }
};

// Ray configuration
struct TraceConfig {
    float tau;             // optical thickness
    float omega;           // single-scattering albedo, omega = sigma_scatter / (sigma_scatter + sigma_absorb
    float max_step;        // longest allowed step, as a fraction of H
    int cells;             // volume cells per side
    float wall_emissivity; // 1 = black walls (absorb everything), <1 = gray walls
    int wall_angle;        // one of the Wall... models above
};

// RAY MOVEMENT

// One Runge-Kutta 4th order step of size dt (in the optical variable t).
// governing equations:
//     d(x,y)/dt = (tx,ty)       d(tx,ty)/dt = -n_ref^2 * a2 * (x,y)
// Move ray by dt
__device__ State rk4_step(const State& s, float dt, const Model& m) {
    const float c = m.n_ref * m.n_ref * m.a2;

    // f(q) = derivative of the state q. We store it in a State as well:
    // (x,y,tx,ty) here means (dx/dt, dy/dt, dtx/dt, dty/dt).
    auto f = [c](const State& q) { return State{q.tx, q.ty, -c * q.x, -c * q.y}; };
    // q + hk, for rk4.
    auto axpy = [](const State& q, const State& k, float h) {return State{q.x + h * k.x, q.y + h * k.y, q.tx + h * k.tx, q.ty + h * k.ty};};

    const State k1 = f(s);
    const State k2 = f(axpy(s, k1, 0.5f * dt)); 
    const State k3 = f(axpy(s, k2, 0.5f * dt)); 
    const State k4 = f(axpy(s, k3, dt)); 

    const float w = dt/6.0f;
    return {s.x  + w * (k1.x  + 2.0f * k2.x  + 2.0f * k3.x  + k4.x),
            s.y  + w * (k1.y  + 2.0f * k2.y  + 2.0f * k3.y  + k4.y),
            s.tx + w * (k1.tx + 2.0f * k2.tx + 2.0f * k3.tx + k4.tx),
            s.ty + w * (k1.ty + 2.0f * k2.ty + 2.0f * k3.ty + k4.ty)};
}

// Move a ray by ds = n*dt, so we guess dt = ds/n, measure the length really travelled (Simpson's rule), and rescale dt until it matches.
__device__ State advance(const State& start, float ds, const Model& m) {
    float dt = ds / m.n(start.x, start.y);
    State end = start;

    for (int iter = 0; iter < 5; ++iter) {
        end = rk4_step(start, dt, m);
        const State mid = rk4_step(start, 0.5f * dt, m);

        const float actual = dt/6.0f * (m.n(start.x, start.y) + 4.0f * m.n(mid.x, mid.y) + m.n(end.x, end.y));
        if (actual <= 0.0f) break;
        if (fabsf(actual - ds)/fmaxf(ds, 1e-20f) < 1e-5f) break;
        dt *= ds / actual; //better guess
    }

    // |T| must equal n(R), correction due to RK4 drift
    const float magT = sqrtf(end.tx*end.tx + end.ty*end.ty);
    if (magT > 0.0f) {
        const float scale = m.n(end.x, end.y)/magT;
        end.tx *= scale;
        end.ty *= scale;
    }
    return end;
}

// WALL

// Did the step a->b leave the square?
struct Hit {
    bool hit;
    int side;       // 0 = left (x=0), 1 = right (x=1), 2 = bottom (y=0), 3 = top (y=1)
    float x, y;     // crossing point
};

// wall detection
__device__ Hit detect(const State& a, const State& b) {
    Hit hit{false, -1, 0, 0};
    float best = 1e30f; // smallest fraction of the step at which a wall is crossed

    const float dx = b.x - a.x;
    const float dy = b.y - a.y;

    #pragma unroll
    for (int side = 0; side < 4; ++side) {
        float wall = (side%2) ? 1.0f : 0.0f; // right/top or left/bottom
        float from = (side<2) ? a.x : a.y;   // x or y coordinate of the start 
        float to = (side<2) ? b.x : b.y;
        float d = to - from;

        // The step must end at or past the wall, moving towards it.
        bool crossed = (side%2) ? (to >= wall && d > 0.0f) : (to <= wall && d < 0.0f);
        if (!crossed) continue;

        float frac = (wall - from) / d; // 0 = start of step, 1 = end
        if (frac >= -1e-6f && frac <= 1.0f + 1e-6f && frac < best) {
            best = frac;
            hit = {true, side, a.x+frac*dx, a.y + frac*dy};
        }
    }

    hit.x = fminf(fmaxf(hit.x, 0.0f), 1.0f);
    hit.y = fminf(fmaxf(hit.y, 0.0f), 1.0f);
    return hit;
}

// wall reflection
__device__ void reflect(int side, int wall_angle, Rng& rng, float& dx, float& dy, float& path_scale) {
    float normal,tangent;   
    path_scale = 1.0f;

    if (wall_angle == WallProjected3D) {
        // 3D Lambertian wall 
        // polar angle theta given by sin^2(theta) ~ uniform(0,1). azimuthal angle phi ~ uniform(0,2pi)
        const float u = (float)(uniform01(rng)); //uniform(0,1)
        const float phi = PI*2*(float)(uniform01(rng));  
        normal = sqrtf(1-u), tangent = sqrtf(u) * cosf(phi);  // cos(theta), sin(theta)*cos(phi)
        path_scale = fmaxf(sqrtf(normal*normal + tangent*tangent), 1e-4f);
        normal /= path_scale;
        tangent /= path_scale;
    } 
    else if (wall_angle == WallIsotropic2D) {
        // Uniform angle over the half-plane facing the domain.
        const float angle = PI*(float)(uniform01(rng));
        normal = sinf(angle);
        tangent = cosf(angle);
    } 
    else {
        // 2D density ~ cos(theta) from the normal, random side.
        const float u = (float)(uniform01(rng));
        normal = sqrtf(1-u);
        tangent = sqrtf(u) * ((uniform01(rng)<0.5) ? -1 : 1);
    }

    // reflect (normal, tangent)
    if (side < 2) {     // left/right wall
        dx = (side == 0) ? normal : -normal;
        dy = tangent;
    } else {            // bottom/top wall
        dx = tangent;
        dy = (side == 2) ? normal : -normal;
    }
}

// RAY TRACE

// Index of the cell (0..n-1) that a coordinate in [0,1] falls into
__device__ int cIdx(float v, int n) {
    return min(max((int)(v*n), 0), n-1);
}

// Follow one ray until it is absorbed. Returns where it ended:
//   0 .. n*n-1          volume cell (row-major: y*n + x)
//   n*n + side*n + k    k-th segment of a wall (side 0..3 as in Hit)
//   -1                  gave up (step limit; should not happen)
// path_scale shortens every free path (used for 3-D projected emission).
__device__ int trace_ray(State s, float path_scale, const Model& m, const TraceConfig& cfg, Rng& rng) {
    int n = cfg.cells;

    // distance to the next interaction ~ exp(1/tau).
    double rem = -log(uniform01(rng))/cfg.tau*path_scale;

    for (int step = 0; step < 2000000; ++step) {
        bool due = rem <= cfg.max_step;
        float ds = min((float)rem, cfg.max_step); 
        State next = advance(s, ds, m);

        // wall hit?
        Hit hit = detect(s, next);
        if (hit.hit) {
            bool reflected = cfg.wall_emissivity<1 && uniform01(rng)>=cfg.wall_emissivity;
            if (reflected) {
                float dx, dy;
                reflect(hit.side, cfg.wall_angle, rng, dx, dy, path_scale);
                float nw = m.n(hit.x, hit.y);
                s = {hit.x, hit.y, nw*dx, nw*dy};
                rem = -log(uniform01(rng)) / cfg.tau*path_scale;
                continue;
            }

            //absorbed
            int segment = cIdx(hit.side < 2 ? hit.y : hit.x, n);
            return n*n + hit.side*n + segment;
        }

        s = next;
        rem -= ds;

        // medium interaction
        if (due || rem <= 1e-12) {
            // absorbed with probability (1-omega)
            if (uniform01(rng) < 1.0 - cfg.omega) {
                return cIdx(s.y, n)*n + cIdx(s.x,n);
            }

            // scatter: turn by a random density ~ 1+cos(delta)
            float delta;
            do {
                delta = PI*2 * (float)(uniform01(rng));
            } while (uniform01(rng) > 0.5*(1+cosf(delta)));

            float n_here = m.n(s.x, s.y), theta = atan2f(s.ty, s.tx) + delta;
            s.tx = n_here*cosf(theta);
            s.ty = n_here*sinf(theta);
            rem = -log(uniform01(rng))/cfg.tau*path_scale;
        }
    }
    return -1;
}
