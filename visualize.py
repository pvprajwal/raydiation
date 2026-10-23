#!/usr/bin/env python3
"""Minimal visualiser for raydiation_output.csv and temperature_field.csv.

By default the CSV is a (cells^2) x (cells^2 + 4*cells) matrix of radiation
distribution factors: row = source cell, column = absorbing volume cell or
wall segment.  This mode makes a 2x2 summary:

  1. fraction of each source cell's rays absorbed in the volume;
  2. wall absorption by wall segment, averaged over all source cells;
  3. destination volume cells for one selected source cell;
  4. wall absorption by wall segment for that selected source cell.

With --temperature it instead plots the CUDA temperature field: the
T/T_w1 profile at x/H = 0.325 against Liu (2006) Table 6, plus the full
2-D temperature field.

Usage:
    python3 visualize.py [results/raydiation_output.csv] \
        [--out results/raydiation_visualization.png] \
        [--source X Y]

    python3 visualize.py --temperature [results/temperature_field.csv] \
        [--benchmark benchmark/liu2006_table6.csv] [--case eps1_omega0] \
        [--x-over-h 0.325] [--out results/temperature_profile.png]

Requires numpy and matplotlib.
"""

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CELL_RE = re.compile(r"cell_y(\d+)_x(\d+)")
WALL_RE = re.compile(r"wall_(left|right|bottom|top)_[xy](\d+)")
SIDES = ("left", "right", "bottom", "top")

try:
    _trapezoid = np.trapezoid
except AttributeError:  # NumPy < 2.0
    _trapezoid = np.trapz


def parse_csv(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    dest_labels = header[1:]
    matrix = np.array([[float(v) for v in row[1:]] for row in rows])

    cell_cols = []
    wall_cols = []
    n = 0
    for col, label in enumerate(dest_labels):
        m = CELL_RE.fullmatch(label)
        if m:
            y, x = map(int, m.groups())
            cell_cols.append((col, y, x))
            n = max(n, y + 1, x + 1)
            continue
        m = WALL_RE.fullmatch(label)
        if m:
            side = SIDES.index(m.group(1))
            seg = int(m.group(2))
            wall_cols.append((col, side, seg))
            n = max(n, seg + 1)
            continue
        raise ValueError(f"unrecognised destination label: {label}")

    src_y = np.empty(len(rows), dtype=int)
    src_x = np.empty(len(rows), dtype=int)
    for i, row in enumerate(rows):
        m = CELL_RE.fullmatch(row[0])
        if not m:
            raise ValueError(f"unrecognised source label: {row[0]}")
        src_y[i], src_x[i] = map(int, m.groups())

    # Rearrange the flat destination columns into (source, y, x) and
    # (source, wall side, segment) arrays.
    cells = np.zeros((len(rows), n, n))
    walls = np.zeros((len(rows), 4, n))
    for col, y, x in cell_cols:
        cells[:, y, x] = matrix[:, col]
    for col, side, seg in wall_cols:
        walls[:, side, seg] = matrix[:, col]

    return cells, walls, src_y, src_x, n, len(dest_labels)


def load_temperature(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    n = 1 + max(max(int(r["cell_y"]), int(r["cell_x"])) for r in rows)
    T = np.zeros((n, n))
    for r in rows:
        T[int(r["cell_y"]), int(r["cell_x"])] = float(r["T_over_Tw1"])
    return T


def load_benchmark(path, case):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return (np.array([float(r["y_over_H"]) for r in rows]),
            np.array([float(r[case]) for r in rows]))


def sample_profile(T, x_over_h, sample_y):
    n = T.shape[0]
    centres = (np.arange(n) + 0.5) / n
    column = np.array([np.interp(x_over_h, centres, T[iy])
                       for iy in range(n)])
    return np.interp(sample_y, centres, column)


def delta_int(T, T_bench):
    return 100.0 * _trapezoid(np.abs(T - T_bench)) / _trapezoid(np.abs(T_bench))


def plot_temperature(path, benchmark, case, x_over_h, out):
    T = load_temperature(path)
    bench_y, bench_T = load_benchmark(benchmark, case)
    T_col = sample_profile(T, x_over_h, bench_y)
    rel = 100.0 * np.abs(T_col - bench_T) / np.abs(bench_T)

    print(f"file: {path}")
    print(f"grid: {T.shape[0]} x {T.shape[1]} cells, profile x/H={x_over_h:.3f}")
    print(f"case: {case}")
    print(f"delta_int={delta_int(T_col, bench_T):.3f}%   "
          f"max|rel|={rel.max():.3f}%   mean|rel|={rel.mean():.3f}%")
    print(f"field mean T^4={np.mean(T**4):.4f}   "
          f"T_top/T_bottom={T_col[-1] / T_col[0]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    ax = axes[0]
    ax.plot(bench_T, bench_y, "o", ms=4, label="Liu (2006)")
    ax.plot(T_col, bench_y, "-", lw=1.5, label="Monte Carlo")
    ax.set(xlabel="T / T_w1", ylabel="y / H",
           title=f"x/H = {x_over_h:.3f}")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    ax = axes[1]
    im = ax.imshow(T, origin="lower", extent=(0, 1, 0, 1), cmap="inferno")
    ax.set(xlabel="x / H", ylabel="y / H", title="T / T_w1 field")
    fig.colorbar(im, ax=ax, label="T / T_w1")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv", nargs="?", default="results/raydiation_output.csv",
        help="input CSV (default: results/raydiation_output.csv)",
    )
    parser.add_argument(
        "--out", default=None,
        help="output PNG (default: matrix mode results/raydiation_visualization.png; "
             "temperature mode results/temperature_profile.png)",
    )
    parser.add_argument(
        "--source", nargs=2, type=int, metavar=("X", "Y"),
        help="source cell to inspect (default: centre cell)",
    )
    parser.add_argument(
        "--temperature", nargs="?", const="results/temperature_field.csv",
        default=None,
        help="plot temperature_field.csv instead of the matrix "
             "(default path when given without a value: "
             "results/temperature_field.csv)",
    )
    parser.add_argument("--benchmark", default="benchmark/liu2006_table6.csv",
                        help="Liu (2006) Table 6 CSV")
    parser.add_argument("--case", default="eps1_omega0",
                        help="benchmark column (default: eps1_omega0)")
    parser.add_argument("--x-over-h", type=float, default=0.325,
                        help="profile column (default: 0.325)")
    args = parser.parse_args()

    if args.temperature:
        out = args.out or "results/temperature_profile.png"
        plot_temperature(args.temperature, args.benchmark, args.case,
                         args.x_over_h, out)
        return

    cells, walls, src_y, src_x, n, ndest = parse_csv(args.csv)
    nsrc = len(src_y)

    if args.source is None:
        sx, sy = n // 2, n // 2
    else:
        sx, sy = args.source
    hits = np.where((src_x == sx) & (src_y == sy))[0]
    if len(hits) != 1:
        raise SystemExit(f"source cell x={sx}, y={sy} not found")
    src = int(hits[0])

    volume = cells.sum(axis=(1, 2))
    wall = walls.sum(axis=(1, 2))
    row_sums = volume + wall

    print(f"file: {args.csv}")
    print(f"source cells: {nsrc}, destinations: {ndest}, grid: {n}x{n}")
    print(f"row sums: min={row_sums.min():.10f} max={row_sums.max():.10f}")
    print(f"mean absorption: volume={volume.mean():.6f}, walls={wall.mean():.6f}")
    print("mean wall absorption by side: "
          + ", ".join(f"{s}={walls[:, i, :].sum(axis=1).mean():.6f}"
                      for i, s in enumerate(SIDES)))
    print(f"selected source: x={sx}, y={sy} "
          f"(volume={volume[src]:.6f}, walls={wall[src]:.6f})")

    volume_grid = np.zeros((n, n))
    volume_grid[src_y, src_x] = volume

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    fig.suptitle(f"Radiation distribution factors — {Path(args.csv).name}",
                 fontsize=14)

    # 1. Where each source cell's rays are absorbed: volume vs walls.
    ax = axes[0, 0]
    im = ax.imshow(volume_grid, origin="lower", extent=(0, n, 0, n),
                   cmap="magma", vmin=volume_grid.min(), vmax=volume_grid.max())
    ax.plot(sx + 0.5, sy + 0.5, marker="*", ms=14, color="cyan",
            markeredgecolor="black", label="selected source")
    ax.set(xlabel="source cell x", ylabel="source cell y",
           title="Fraction absorbed in volume (per source cell)")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, label="fraction of source rays")

    # 2. Wall absorption by segment, averaged over all source cells.
    ax = axes[0, 1]
    mean_walls = walls.mean(axis=0)
    for side, label in enumerate(SIDES):
        ax.plot(np.arange(n), mean_walls[side], marker=".", label=label)
    ax.set(xlabel="wall segment index", ylabel="mean fraction per source cell",
           title="Wall absorption by segment (all sources)")
    ax.grid(alpha=0.25)
    ax.legend()

    # 3. Destination volume cells for the selected source cell.
    ax = axes[1, 0]
    dest = cells[src].reshape(n, n)
    im = ax.imshow(dest, origin="lower", extent=(0, 1, 0, 1),
                   cmap="viridis", vmin=0, vmax=dest.max())
    ax.plot((sx + 0.5) / n, (sy + 0.5) / n, marker="*", ms=14, color="red",
            markeredgecolor="white", label="source")
    ax.set(xlabel="destination cell x", ylabel="destination cell y",
           title=f"Destination cells, source x={sx}, y={sy}")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, label="fraction of source rays")

    # 4. Wall absorption by segment for the selected source cell.
    ax = axes[1, 1]
    for side, label in enumerate(SIDES):
        ax.plot(np.arange(n), walls[src, side], marker=".", label=label)
    ax.set(xlabel="wall segment index", ylabel="fraction of source rays",
           title=f"Wall absorption, source x={sx}, y={sy}")
    ax.grid(alpha=0.25)
    ax.legend()

    out = Path(args.out or "results/raydiation_visualization.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
