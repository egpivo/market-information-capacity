#!/usr/bin/env python3
"""Animate the participation curve from one trader to one million.

Reads only `results/participation_scale.csv`, which the Rust binary writes.
Every plotted value is the `analytic_mse` / `analytic_floor` column; nothing is
recomputed here except the share of the interpretation-noise budget already
averaged away, which is `1 - 1/N_T` and is stated in the caption.

Usage:
    python3 scripts/make_participation_gif.py [--results results] [--figures figures]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from PIL import Image
except ImportError as exc:  # pragma: no cover - environment dependent
    sys.exit(f"animation requires pandas, matplotlib and pillow ({exc})")

BG = "#F7F6F3"
TEXT = "#2B2B2B"
MUTED = "#5F6368"
GRID = "#D9D7D1"

# Same semantic mapping as every other figure: how thick the information budget is.
THIN, MODERATE, RICH = "#8A6799", "#4C6A91", "#6A8F73"

# Frames that hold longer, because they are the ones worth reading.
HOLD = {1: 900, 10: 950, 100: 950, 2500: 950, 1_000_000: 1800}
STEP_MS = 300

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Source Sans 3", "Arial", "Helvetica", "DejaVu Sans"],
    "text.color": TEXT, "axes.labelcolor": TEXT, "axes.edgecolor": "#B9B6AE",
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})


def thousands(n: int) -> str:
    return f"{n:,}"


def draw_frame(curve: pd.DataFrame, budgets: list[int], colours: dict, upto: int,
               ymin: float, ymax: float) -> Image.Image:
    fig, ax = plt.subplots(figsize=(9.6, 5.6), dpi=110)

    for k in budgets:
        cell = curve[curve["k_sources"] == k].sort_values("n_traders")
        colour = colours[k]
        floor = float(cell["analytic_floor"].iloc[0])

        # Floors are always visible, so the reader knows where each curve is headed.
        ax.axhline(floor, color=colour, linestyle="--", linewidth=1.2, alpha=0.45)

        shown = cell[cell["n_traders"] <= upto]
        if len(shown):
            ideal = shown[shown["full_coverage_assumed"]]
            covered = shown[~shown["full_coverage_assumed"]]
            if len(ideal):
                bridge = pd.concat([ideal, covered.head(1)])
                ax.plot(bridge["n_traders"], bridge["analytic_mse"],
                        color=colour, linewidth=2.0, linestyle=":", alpha=0.85)
            if len(covered):
                ax.plot(covered["n_traders"], covered["analytic_mse"],
                        color=colour, linewidth=2.8, solid_capstyle="round")
            head = shown.iloc[-1]
            ax.plot(head["n_traders"], head["analytic_mse"], "o",
                    color=colour, markersize=11, markeredgecolor=BG, markeredgewidth=2,
                    zorder=6)

        # Static labels past the right edge of the data: nothing moves between frames.
        # Nudge apart: the two richest budgets have floors only 0.04 apart.
        offset = {budgets[0]: 0, budgets[1]: 7, budgets[-1]: -8}.get(k, 0)
        ax.annotate(f"K = {k}", xy=(1.35e6, floor), xytext=(6, offset),
                    textcoords="offset points", color=colour, fontsize=14.5,
                    va="center", ha="left", fontweight="medium")

    ax.set_xscale("log")
    ax.set_xlim(0.75, 9.0e6)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Traders $N_T$ (log scale)", fontsize=15)
    ax.set_ylabel("Price error MSE($P^{OC}$, V)", fontsize=15)
    ax.tick_params(labelsize=13)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_title("Adding traders to a fixed information budget", fontsize=19, pad=16)

    captured = 1.0 - 1.0 / upto
    removable = 0.64 / upto
    # The upper-right quadrant stays empty at every frame, so the readout can live
    # there without ever colliding with a curve or a floor line.
    ax.text(0.40, 0.96, f"$N_T$ = {thousands(upto)}", transform=ax.transAxes,
            fontsize=27, color=TEXT, va="top", ha="left")
    ax.text(0.40, 0.845, f"{captured * 100:.6g}% of what more traders can ever remove",
            transform=ax.transAxes, fontsize=15.5, color=TEXT, va="top", ha="left")
    ax.text(0.40, 0.775, "is already gone", transform=ax.transAxes,
            fontsize=15.5, color=TEXT, va="top", ha="left")
    left = f"{removable:.4f}" if removable >= 1e-3 else f"{removable:.1e}"
    ax.text(0.40, 0.685, f"still removable by participation:  {left}",
            transform=ax.transAxes, fontsize=13.5, color=MUTED, va="top", ha="left")
    ax.text(0.40, 0.625, "dashed: the floor its sources allow",
            transform=ax.transAxes, fontsize=13, color=MUTED, va="top", ha="left")

    fig.tight_layout()
    fig.canvas.draw()
    image = Image.frombytes(
        "RGBA", fig.canvas.get_width_height(), bytes(fig.canvas.buffer_rgba())
    ).convert("RGB")
    plt.close(fig)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    args = parser.parse_args()

    source = args.results / "participation_scale.csv"
    if not source.exists():
        print(f"missing {source}; run: cargo run --release -- scale-participation",
              file=sys.stderr)
        return 1

    curve = pd.read_csv(source)
    budgets = sorted(curve["k_sources"].unique())
    colours = dict(zip(budgets, [THIN, MODERATE, RICH]))
    steps = sorted(curve["n_traders"].unique())

    lo = float(curve["analytic_floor"].min())
    hi = float(curve["analytic_mse"].max())
    pad = 0.08 * (hi - lo)
    ymin, ymax = lo - pad, hi + pad

    frames = [draw_frame(curve, budgets, colours, n, ymin, ymax) for n in steps]
    durations = [HOLD.get(n, STEP_MS) for n in steps]
    print(f"rendered {len(frames)} frames")

    args.figures.mkdir(parents=True, exist_ok=True)
    gif = args.figures / "participation_saturation.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    print(f"wrote {gif} ({gif.stat().st_size / 1_000_000:.2f} MB, "
          f"{sum(durations) / 1000:.1f} s per loop)")

    # A still of the first frame, so the axes read even where GIFs do not autoplay.
    poster = args.figures / "participation_saturation_first_frame.png"
    frames[0].save(poster)
    print(f"wrote {poster}")

    mp4 = args.figures / "participation_saturation.mp4"
    tmp = args.figures / "_frames"
    tmp.mkdir(exist_ok=True)
    try:
        for i, (frame, ms) in enumerate(zip(frames, durations)):
            for repeat in range(max(1, round(ms / STEP_MS))):
                frame.save(tmp / f"f{i:03d}_{repeat}.png")
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", f"{1000 / STEP_MS:.4f}",
             "-pattern_type", "glob", "-i", str(tmp / "*.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(mp4)],
            check=True, capture_output=True)
        print(f"wrote {mp4} ({mp4.stat().st_size / 1_000_000:.2f} MB)")
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"skipped mp4 ({exc.__class__.__name__}); the GIF is the deliverable",
              file=sys.stderr)
    finally:
        for f in tmp.glob("*.png"):
            f.unlink()
        tmp.rmdir()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
