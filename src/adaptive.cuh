#pragma once

#include "model.cuh"

struct AdaptiveConfig {
    float tol = 1e-6f;     // local position/direction error
    float min_step = 1e-4f;      // minimum geometric step
    float max_step = 0.08f;       // maximum geometric step
    float init_step = 0.02f;      // initial step of each free flight
    float boundary_layer = 0.02f;   // resolve this layer at the v1 step
    float boundary_step = 0.005f;   // cap inside the boundary layer
    float safety = 0.9f;
    float min_factor = 0.2f;
    float max_factor = 4.0f;
    int max_attempts = 10;
    float collision_rel_tol = 1e-5f;
};

// ds = n dt, integrated with Simpson's rule over the trial step.
__device__ float simpson_arc_length(float dt, const State& a, const State& mid, const State& b, const Model& m) {
    return dt/6.0f * (m.n(a.x, a.y) + 4.0f*m.n(mid.x, mid.y) + m.n(b.x, b.y));
}

// Straight-line distance to the first wall along the optical direction. This
// is only a step limiter; detect() still finds the actual crossing.
__device__ float distance_to_boundary(const State& s) {
    float mag = sqrtf(s.tx*s.tx + s.ty*s.ty);
    if (!(mag > 1e-20f)) return 0.0f;

    float vx = s.tx/mag, vy = s.ty/mag;
    float d = 1e30f;
    if (vx > 1e-12f) d = fminf(d, (1.0f - s.x)/vx);
    if (vx < -1e-12f) d = fminf(d, (0.0f - s.x)/vx);
    if (vy > 1e-12f) d = fminf(d, (1.0f - s.y)/vy);
    if (vy < -1e-12f) d = fminf(d, (0.0f - s.y)/vy);
    return fmaxf(d, 0.0f);
}

struct AdaptiveStepResult {
    State end;
    float distance = 0.0f;
    float next_step = 0.0f;
    bool reached_target = false;
};

__device__ __forceinline__ AdaptiveStepResult adaptive_step(const State& start, float target_distance, bool collision, const Model& m, const AdaptiveConfig& ac) {
    AdaptiveStepResult out{};
    float n0 = m.n(start.x, start.y);
    float dt_min = ac.min_step/fmaxf(n0, 1e-6f);
    float dt = target_distance/fmaxf(n0, 1e-6f);
    bool require_target = collision;
    float best_dist_err = 1e30f;
    State best_end = start;

    for (int attempt = 0; attempt < ac.max_attempts; ++attempt) {
        State mid = rk4_step(start, 0.5f*dt, m);
        State full = rk4_step(start, dt, m);
        State half = rk4_step(mid, 0.5f*dt, m);
        float ds = simpson_arc_length(dt, start, mid, half, m);

        float pos_err = fmaxf(fabsf(full.x - half.x), fabsf(full.y - half.y));
        float dir_err = fmaxf(fabsf(full.tx - half.tx), fabsf(full.ty - half.ty))/fmaxf(n0, 1.0f);
        float err = fmaxf(pos_err, dir_err);

        float dist_err = 0.0f;
        if (require_target && ds > 0.0f) {
            dist_err = fabsf(ds - target_distance)/target_distance;
            if (dist_err < best_dist_err) {
                best_dist_err = dist_err;
                best_end = half;
            }
            if (dist_err > ac.collision_rel_tol) {
                dt *= target_distance/ds;
                if (dt < dt_min) dt = dt_min;
                continue;
            }
        }

        if (err > ac.tol && dt > dt_min) {
            float factor = ac.safety*powf(ac.tol/fmaxf(err, 1e-30f), 0.2f);
            factor = fminf(fmaxf(factor, ac.min_factor), 1.0f);
            dt *= factor;
            if (dt < dt_min) dt = dt_min;
            // A shorter step is no longer the collision step.
            require_target = false;
            continue;
        }

        out.end = half;
        out.distance = ds;
        float factor = ac.safety*powf(ac.tol/fmaxf(err, 1e-30f), 0.2f);
        factor = fminf(fmaxf(factor, ac.min_factor), ac.max_factor);
        out.next_step = ds*factor;
        out.reached_target = collision ? (require_target && dist_err <= ac.collision_rel_tol) : true;
        return out;
    }

    // keep the closest collision candidate, otherwise use the fixed-step advance()
    if (collision && best_dist_err < 1e30f) {
        out.end = best_end;
        out.distance = target_distance;
        out.next_step = target_distance;
        out.reached_target = true;
    } else {
        out.end = advance(start, target_distance, m);
        out.distance = target_distance;
        out.next_step = target_distance;
        out.reached_target = true;
    }
    return out;
}

// Adaptive trace 
__device__ __forceinline__ int trace_ray_adaptive(State s, float path_scale, const Model& m, const TraceConfig& cfg, const AdaptiveConfig& ac, Rng& rng) {
    int n = cfg.cells;

    double remaining = -log(uniform01(rng))/cfg.tau*path_scale;
    float h = ac.init_step;

    for (int step = 0; step < 2000000; ++step) {
        float t_wall = distance_to_boundary(s);
        float boundary_limit = fmaxf(ac.boundary_step, t_wall - ac.boundary_layer);

        
        float d_perp = fminf(fminf(s.x, 1.0f - s.x), fminf(s.y, 1.0f - s.y));
        if (d_perp < ac.boundary_layer) boundary_limit = ac.boundary_step;

        bool collision_due = remaining <= 1.25*(double)h && remaining <= (double)boundary_limit;
        float ds_limit;
        bool limited;
        if (collision_due) {
            ds_limit = (float)remaining;
            limited = true;
        } else {
            ds_limit = fminf(h, boundary_limit);
            limited = boundary_limit < h;
        }
        if (!collision_due && ds_limit < ac.min_step) ds_limit = ac.min_step;

        AdaptiveStepResult sr = adaptive_step(s, ds_limit, collision_due, m, ac);

        Hit hit = detect(s, sr.end);
        if (hit.hit) {
            bool reflected = cfg.wall_emissivity < 1.0f && uniform01(rng) >= cfg.wall_emissivity;
            if (reflected) {
                float dx, dy;
                reflect(hit.side, cfg.wall_angle, rng, dx, dy, path_scale);
                float nw = m.n(hit.x, hit.y);
                s = {hit.x, hit.y, nw*dx, nw*dy};
                remaining = -log(uniform01(rng))/cfg.tau*path_scale;
                h = ac.init_step;
                continue;
            }

            int segment = cIdx(hit.side < 2 ? hit.y : hit.x, n);
            return n*n + hit.side*n + segment;
        }

        s = sr.end;
        remaining -= sr.distance;

        bool reached_collision = collision_due && sr.reached_target;
        if (reached_collision || remaining <= 1e-12) {
            if (uniform01(rng) < 1.0 - cfg.omega) {
                return cIdx(s.y, n)*n + cIdx(s.x, n);
            }

            float delta;
            do {
                delta = PI*2*(float)uniform01(rng);
            } while (uniform01(rng) > 0.5*(1.0 + cosf(delta)));

            float n_here = m.n(s.x, s.y), theta = atan2f(s.ty, s.tx) + delta;
            s.tx = n_here*cosf(theta);
            s.ty = n_here*sinf(theta);
            remaining = -log(uniform01(rng))/cfg.tau*path_scale;
            h = ac.init_step;
        } else {
            float factor = sr.distance > 0.0f ? sr.next_step/sr.distance : 1.0f;
            if (limited) {
                h = fminf(fmaxf(h*factor, ac.min_step), ac.max_step);
            } else {
                h = fminf(fmaxf(sr.next_step, ac.min_step), ac.max_step);
            }
        }
    }
    return -1;
}
