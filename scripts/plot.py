#!/usr/bin/env python3
"""Render figures from Rust-generated result files.

This script is the only Python in the repository and it does exactly one job:
turn canonical CSVs written by the Rust binary into PNGs. It runs no
simulation, defines no model parameter, and computes no canonical statistic --
every number it draws, including every analytic floor, is read from a column
that Rust already wrote.

Usage:
    python3 scripts/plot.py [--results results] [--figures figures]
    python3 scripts/plot.py --assets assets     # also refresh the README figures

Requires pandas and matplotlib. If they are unavailable the Rust scientific
pipeline still runs and still produces every canonical result file; only the
figures are skipped.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.lines import Line2D
except ImportError as exc:  # pragma: no cover - environment dependent
    sys.exit(
        f"plotting requires pandas and matplotlib ({exc}); "
        "the Rust pipeline does not depend on them"
    )

# A calm, colour-blind-safe palette. Curves and their floors share a colour.
PALETTE = ["#1b5e7e", "#b4632a", "#3f7a4f", "#7a4f8c", "#8c6d1f", "#6b6b6b"]

# One semantic mapping for information budget, shared across every figure.
THIN = "#8A6799"      # muted plum  — smallest K
MODERATE = "#4C6A91"  # steel blue  — middle K
RICH = "#6A8F73"      # sage green  — largest K
BG = "white"
GRID = "#d9d9d9"
MUTED = "#5a5a5a"
REF = "#7a7a7a"
GRID_KW = dict(color="#d9d9d9", linewidth=0.6)

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 220,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.edgecolor": "#4a4a4a",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
    }
)


def _finish(fig, path: Path, caption: str, lines: int = 2) -> None:
    width = int(fig.get_size_inches()[0] * 15)
    wrapped = "\n".join(textwrap.wrap(caption, width=width))
    fig.text(0.012, 0.012, wrapped, fontsize=8, color="#5a5a5a", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.035 + 0.028 * lines, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def plot_traders_vs_information(results: Path, figures: Path) -> None:
    """Simulated MSE against trader count, with each K curve's analytic floor.

    The floor is read from the `analytic_floor` column; it is not recomputed
    here. Each K uses one colour for both its simulated curve and its dashed
    floor, so the pairing is unambiguous.
    """
    grid = pd.read_csv(results / "traders_vs_sources.csv")
    shown = [k for k in (2, 10, 50) if k in set(grid["k_sources"])]
    if not shown:
        shown = sorted(grid["k_sources"].unique())[:3]

    fig, ax = plt.subplots(figsize=(9, 5))
    for colour, k in zip(PALETTE, shown):
        cell = grid[grid["k_sources"] == k].sort_values("n_traders")
        floor = cell["analytic_floor"].iloc[0]
        ax.errorbar(
            cell["n_traders"],
            cell["mse"],
            yerr=cell["mc_se"],
            marker="o",
            markersize=4.5,
            linewidth=1.6,
            elinewidth=1.0,
            capsize=2.5,
            color=colour,
            ecolor=colour,
            label=f"K = {k}",
        )
        ax.axhline(floor, color=colour, linestyle="--", linewidth=1.2, alpha=0.8)
        ax.annotate(
            f"K = {k}\nfloor {floor:.3f}",
            xy=(cell["n_traders"].max(), floor),
            xytext=(10, -2),
            textcoords="offset points",
            fontsize=8.5,
            color=colour,
            ha="left",
            va="center",
        )

    ax.set_xscale("log")
    ax.set_xlim(right=float(grid["n_traders"].max()) * 3.4)
    ax.set_xlabel("Traders $N_T$ (log scale)")
    ax.set_ylabel("MSE of onchain price against latent value")
    ax.set_title("More traders do not mean more information")
    ax.grid(axis="y", **GRID_KW)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", ncol=len(shown))
    ax.margins(y=0.12)

    rho = grid["rho_s"].iloc[0]
    sigma = grid["sigma_s"].iloc[0]
    nu = grid["interpretation_sigma"].iloc[0]
    _finish(
        fig,
        figures / "traders_vs_information.png",
        f"Markers: simulated MSE with Monte Carlo standard errors. Dashed: analytic floor "
        f"sigma_s^2 [rho_s + (1 - rho_s)/K], drawn in the same colour as its curve. "
        f"rho_s={rho}, sigma_s={sigma}, interpretation noise={nu}. "
        f"Source: results/traders_vs_sources.csv.",
        lines=3,
    )


def plot_same_price(results: Path, figures: Path) -> None:
    """Conditional distribution of the latent value given the same price band."""
    rows = pd.read_csv(results / "same_price.csv")
    rows = rows.iloc[::-1].reset_index(drop=True)  # widest at the bottom
    y = range(len(rows))

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.axvline(0, color="#4a4a4a", linestyle="--", linewidth=0.9)
    for i, row in rows.iterrows():
        colour = PALETTE[i % len(PALETTE)]
        ax.plot(
            [row["q05_v"], row["q95_v"]],
            [i, i],
            color=colour,
            linewidth=3.0,
            solid_capstyle="round",
            alpha=0.85,
        )
        ax.plot(row["mean_v"], i, "o", color=colour, markersize=7)
        ax.annotate(
            f"sd {row['sd_v']:.3f}    K={int(row['k_sources'])}, "
            f"$\\rho_s$={row['rho_s']}    {int(row['accepted_n']):,} accepted",
            xy=(row["mean_v"], i),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8.5,
            color="#4a4a4a",
        )

    ax.set_yticks(list(y))
    ax.set_yticklabels(rows["world"])
    ax.set_ylim(-0.75, len(rows) - 0.35)
    band = rows["band"].iloc[0]
    ax.set_xlabel(f"Latent value $V$ given $|P_{{OC}}| < {band}$")
    ax.set_title("Similar onchain prices can carry very different information")
    ax.grid(axis="x", **GRID_KW)
    ax.set_axisbelow(True)
    ax.margins(x=0.08)

    _finish(
        fig,
        figures / "same_price.png",
        "Dot: conditional mean of V. Bar: 5th to 95th percentile. Every world is "
        "conditioned on the same price band and carries the same trader count, so the "
        "difference in width is a difference in information, not in participation. "
        "Source: results/same_price.csv.",
        lines=2,
    )


def plot_official_boundary(results: Path, figures: Path) -> None:
    """Pre- and post-boundary price MSE by world (diagnostic)."""
    rows = pd.read_csv(results / "worlds.csv")
    x = range(len(rows))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(
        [i - width / 2 for i in x],
        rows["pre_price_mse"],
        width,
        color=PALETTE[0],
        label="before official information",
    )
    ax.bar(
        [i + width / 2 for i in x],
        rows["post_price_mse"],
        width,
        color=PALETTE[1],
        label="after official information",
    )
    for i, row in rows.iterrows():
        ax.annotate(
            f"mean |revision| {row['mean_abs_revision']:.2f}",
            xy=(i, max(row["pre_price_mse"], row["post_price_mse"])),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color="#4a4a4a",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [
            f"{r['world']}\nK={int(r['k_sources'])}, $\\rho_s$={r['rho_s']}"
            for _, r in rows.iterrows()
        ]
    )
    ax.set_ylabel("MSE of price against latent value")
    ax.set_title("Thin information prices badly, then moves a long way")
    ax.grid(axis="y", **GRID_KW)
    ax.set_axisbelow(True)
    ax.margins(y=0.16)
    ax.legend(loc="upper left")

    _finish(
        fig,
        figures / "official_boundary.png",
        "The official revision is a fixed-weight rule, not a Bayesian update, so an "
        "already-accurate price can get worse. Source: results/worlds.csv.",
        lines=2,
    )


def plot_phase_slices(results: Path, figures: Path) -> None:
    """Relative gap to the analytic floor across parameter slices (diagnostic)."""
    rows = pd.read_csv(results / "information_phase_slices.csv")
    slices = list(dict.fromkeys(rows["slice"]))
    if not slices:
        return
    vmin, vmax = rows["gap_to_floor"].min(), rows["gap_to_floor"].max()

    fig, axes = plt.subplots(
        1, len(slices), figsize=(4.6 * len(slices), 4.2), sharex=True, sharey=True
    )
    axes = [axes] if len(slices) == 1 else list(axes)
    image = None
    for ax, label in zip(axes, slices):
        cell = rows[rows["slice"] == label]
        table = cell.pivot(
            index="clientele_bias", columns="sigma_s", values="gap_to_floor"
        )
        image = ax.imshow(
            table.values,
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            extent=(
                float(table.columns.min()),
                float(table.columns.max()),
                float(table.index.min()),
                float(table.index.max()),
            ),
        )
        k = int(cell["k_sources"].iloc[0])
        rho = cell["rho_s"].iloc[0]
        ax.set_title(f"K={k}, $\\rho_s$={rho}")
        ax.set_xlabel("source error scale $\\sigma_s$")
    axes[0].set_ylabel("clientele tilt")
    if image is not None:
        fig.colorbar(image, ax=axes, label="relative gap to analytic floor")
    fig.suptitle("Distance above the information floor, common scale")

    figures.mkdir(parents=True, exist_ok=True)
    path = figures / "information_phase_slices.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_participation_scale(results: Path, figures: Path) -> None:
    """Price error from one trader to one million, with analytic floors.

    The analytic curve is `analytic_mse`, written by the simulation; the dashed
    floors are its `analytic_floor` column. Monte Carlo points are drawn only
    where the simulation validated the curve.
    """
    curve = pd.read_csv(results / "participation_scale.csv")
    budgets = sorted(curve["k_sources"].unique())
    colours = dict(zip(budgets, [THIN, MODERATE, RICH]))

    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    for k in budgets:
        cell = curve[curve["k_sources"] == k].sort_values("n_traders")
        colour = colours.get(k, PALETTE[0])
        floor = cell["analytic_floor"].iloc[0]

        # Below one trader per source the model still credits the price with the
        # full K-source aggregate. Draw that stretch dotted: it is an
        # idealisation, not a claim about markets with fewer traders than
        # sources.
        idealised = cell[cell["full_coverage_assumed"]]
        covered = cell[~cell["full_coverage_assumed"]]
        if len(idealised):
            bridge = pd.concat([idealised, covered.head(1)])
            ax.plot(bridge["n_traders"], bridge["analytic_mse"],
                    color=colour, linewidth=1.8, linestyle=":", alpha=0.85)
        ax.plot(covered["n_traders"], covered["analytic_mse"],
                color=colour, linewidth=2.2, label=f"K = {k}")
        ax.axhline(floor, color=colour, linestyle="--", linewidth=1.3, alpha=0.85)
        # Stagger the labels: the two lowest floors sit close together.
        below = k == budgets[-1]
        ax.annotate(f"K = {k} floor  {floor:.3f}",
                    xy=(cell["n_traders"].max(), floor),
                    xytext=(14, -6 if below else 5),
                    textcoords="offset points", va="top" if below else "bottom",
                    ha="left", fontsize=13, color=colour)

        mc = cell.dropna(subset=["mc_mse"])
        ax.errorbar(mc["n_traders"], mc["mc_mse"], yerr=mc["mc_se"],
                    fmt="o", markersize=6, color=colour, ecolor=colour,
                    elinewidth=1.2, capsize=3,
                    markerfacecolor=BG, markeredgewidth=1.6, zorder=5)

    ax.set_xscale("log")
    ax.set_xlim(0.7, float(curve["n_traders"].max()) * 6)
    ax.set_xlabel("Traders $N_T$ (log scale)")
    ax.set_ylabel("Price error MSE($P^{OC}$, V)")
    ax.set_title("More traders eventually hit an information floor", pad=14)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    idealised_proxy = Line2D([0], [0], color=MUTED, linestyle=":", linewidth=1.8)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [idealised_proxy], labels + ["$N_T$ < K: full coverage assumed"],
              loc="upper right", ncol=2)
    ax.margins(y=0.12)

    fig.tight_layout()
    path = figures / "traders_vs_information_scale.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def plot_marginal_gain(results: Path, figures: Path) -> None:
    """Informational return to doubling participation.

    The gain is identical across source budgets under this model — it is
    `sigma_nu^2 / (2 N_T)`, in which K does not appear — so one curve is drawn
    rather than three overlapping ones.
    """
    gain = pd.read_csv(results / "marginal_information_gain.csv")
    budgets = sorted(gain["k_sources"].unique())
    reference = gain[gain["k_sources"] == budgets[0]].sort_values("n_traders")
    identical = all(
        abs(gain[gain["k_sources"] == k].sort_values("n_traders")["analytic_gain"].values
            - reference["analytic_gain"].values).max() < 1e-15
        for k in budgets
    )

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.plot(reference["n_traders"], reference["analytic_gain"],
            color=MODERATE, linewidth=2.2,
            label="all source budgets" if identical else f"K = {budgets[0]}")
    if not identical:
        for k in budgets[1:]:
            cell = gain[gain["k_sources"] == k].sort_values("n_traders")
            ax.plot(cell["n_traders"], cell["analytic_gain"],
                    linewidth=2.0, label=f"K = {k}")

    resolved = gain[gain["mc_gain_resolved"] == True]  # noqa: E712
    unresolved = gain[gain["mc_gain_resolved"] == False]  # noqa: E712
    if len(resolved):
        ax.errorbar(resolved["n_traders"], resolved["mc_gain"], yerr=resolved["mc_gain_se"],
                    fmt="o", markersize=6, color=MODERATE, ecolor=MODERATE,
                    elinewidth=1.2, capsize=3, markerfacecolor=BG, markeredgewidth=1.6,
                    zorder=5, label="simulated, resolved")
    if len(unresolved):
        se = float(unresolved["mc_gain_se"].median())
        ax.axhline(3 * se, color=REF, linestyle="-.", linewidth=1.2)
        ax.annotate("below here the simulation cannot separate the gain from zero",
                    xy=(float(gain["n_traders"].max()), 3 * se), xytext=(0, -10),
                    textcoords="offset points", fontsize=12, color=MUTED,
                    ha="right", va="top")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Current traders $N_T$ (log scale)")
    ax.set_ylabel("MSE removed by doubling participation")
    ax.set_title("The return to doubling participation falls as $1/N_T$", pad=14)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left")

    fig.tight_layout()
    path = figures / "marginal_information_gain.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


PLOTS = {
    "traders_vs_sources.csv": plot_traders_vs_information,
    "same_price.csv": plot_same_price,
    "worlds.csv": plot_official_boundary,
    "information_phase_slices.csv": plot_phase_slices,
}

# Rendered only when the participation-scale experiment has been run.
OPTIONAL_PLOTS = {
    "participation_scale.csv": plot_participation_scale,
    "marginal_information_gain.csv": plot_marginal_gain,
}

# The two figures the README embeds, committed under assets/ so a reader does not
# need a Python environment to see them. Everything else in figures/ is a
# regenerated diagnostic.
README_FIGURES = {
    "traders_vs_information.png": "traders-vs-information.png",
    "same_price.png": "same-price.png",
}


def refresh_assets(figures: Path, assets: Path) -> None:
    """Copy the README figures out of the generated set."""
    assets.mkdir(parents=True, exist_ok=True)
    for source_name, asset_name in README_FIGURES.items():
        source = figures / source_name
        if not source.exists():
            print(f"skipped {asset_name}: {source} not generated", file=sys.stderr)
            continue
        target = assets / asset_name
        target.write_bytes(source.read_bytes())
        print(f"wrote {target}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    parser.add_argument(
        "--assets",
        type=Path,
        default=None,
        help="also refresh the committed README figures in this directory",
    )
    args = parser.parse_args()

    missing = [name for name in PLOTS if not (args.results / name).exists()]
    if missing:
        print(
            "missing result files: "
            + ", ".join(missing)
            + "\nrun: cargo run --release -- publication",
            file=sys.stderr,
        )
        return 1

    for name, plot in PLOTS.items():
        plot(args.results, args.figures)
    for name, plot in OPTIONAL_PLOTS.items():
        if (args.results / name).exists():
            plot(args.results, args.figures)
    if args.assets is not None:
        refresh_assets(args.figures, args.assets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
