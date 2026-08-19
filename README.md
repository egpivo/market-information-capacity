# market-information-capacity

Deterministic Rust toolkit for measuring how finite independent information constrains market price informativeness as trader participation grows.

It does **not** treat `traders ↑` as `information ↑`. Trader count `N_T` and source budget `K` are separate state variables. Belief formation receives a `SourceSet`, never the latent value, so no participation model can manufacture fundamental information — the constraint is a trait signature, not a convention. Price error converges to `σ_s²[ρ_s + (1−ρ_s)/K] + b²`, in which `N_T` does not appear.

Pure simulation. No market data, no calibration to any real venue, no welfare, no optimal participant count.

Model, validation and claim boundaries: `research/MODEL.md`, `research/VALIDATION.md`, `research/CLAIMS.md` (gitignored local notes).

## Prerequisites

- Rust 1.85+ (`rustup`, stable toolchain)
- Cargo
- Python 3 with pandas and matplotlib, for figures only

## Installation

```bash
git clone <repo-url> market-information-capacity
cd market-information-capacity
cargo build --release
```

## Usage

```bash
cargo test --release

cargo run --release -- validate --verbose
cargo run --release -- publication --config configs/publication.toml

cargo run --release -- asymptote
cargo run --release -- traders-vs-sources
cargo run --release -- same-price
cargo run --release -- worlds
cargo run --release -- sensitivity
cargo run --release -- scale-participation --config configs/publication.toml

cargo run --release -- worlds --config configs/quick.toml --seed 20260915 --output-dir results

python3 scripts/plot.py
python3 scripts/make_participation_gif.py
```

Exit codes: `0` on a successful run **or** a passing gate; `1` when a validation check fails or a config/IO error occurs. `validate` is the only command that can fail on economics rather than mechanics.

`validate` prints nine checks and the evidence behind each with `--verbose`. A `PASS` line without a measured number is not the output format — every check reports what it measured.

`./scripts/reproduce.sh` runs fmt, clippy, tests, `validate`, `publication`, `scale-participation` and the figures in one pass. Publication scale takes a few seconds on ten cores; a million-trader cell costs what a one-trader cell costs, because the trader count enters through a variance rather than a million objects.

Experiments live under `configs/`: `publication.toml` states every canonical parameter and is byte-equivalent to `Config::publication()`, enforced by `tests/config_contract.rs`; `quick.toml` is the same experiment set at reduced Monte Carlo scale. Three parameter regimes — Informative, Selected Clientele, Fast Follower — are configuration, never separate code paths. Results land in `results/` and figures in `figures/`, both regenerated in about a second and neither tracked.

Every simulated value is written next to its closed form, so `analytic_floor` and `analytic_mse` are columns in the output rather than curves fitted at plot time.

Master seed `20260915`. Seeds derive from a pure hash of `(experiment, cell, block, batch)`, so parallelism changes runtime and nothing else; `tests/reproducibility.rs` asserts bitwise reproduction.

Claim gate for article numbers: `cargo run --release -- validate` and `research/CLAIMS.md`.

MIT — see `LICENSE`.
