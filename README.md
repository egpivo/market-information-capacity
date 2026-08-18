# Market Information Capacity

**More traders do not necessarily mean more information.**

A market can contain thousands of participants while those participants process
only a small, correlated set of underlying signals. Once those signals are
already represented in the price, adding traders can increase participation
without proportionally increasing what the market knows. Participation and
information are separate quantities, and only one of them is what the price is
made of.

This repository is a pure-Rust research tool that makes that statement precise,
simulates it, and refuses to publish a result that fails to hold.

## The result

Let `V` be the latent value a market is trying to price, let there be `K`
fundamental information sources about it, and let the errors in those sources be
correlated with coefficient `rho_s` and scale `sigma_s`. As the trader
population grows, the mean squared error of the market price against `V`
converges to

```
MSE_inf(K, rho_s) = sigma_s^2 [ rho_s + (1 - rho_s) / K ]
```

* `K` — the number of fundamental sources. More sources lower the floor.
* `rho_s` — how much error the sources share. More correlation raises the floor,
  and no amount of `K` removes the shared part: as `K` grows the floor tends to
  `sigma_s^2 * rho_s`, not to zero.
* `sigma_s` — how noisy each source is.
* The trader count `N_T` **does not appear**.

The independent-source case is the familiar `sigma_s^2 / K`. The general case is
the one that matters here, because real information sources are correlated:
analysts read the same filings, the same round of coverage, the same handful of
primary reports.

Traders are not absent from the model — they are just doing something else. Each
trader reads an existing source and adds interpretation noise; averaging over
more traders averages that noise away at rate `sigma_nu^2 / N_T`. That term
vanishes. The floor does not.

What the simulation shows, at `K = 10`, `rho_s = 0.5`, with 100,000 realizations
per arm:

| Starting point | Doubling the traders buys | Doubling the sources buys |
|---|---:|---:|
| 100 traders | 0.006 | 0.025 |
| 1,000 traders | 0.001 | 0.027 |
| 5,000 traders | 0.003 | 0.022 |

The trader column is Monte Carlo noise around an exact value of 0.0032, 0.00032
and 0.000064. The source column is a real 0.025 every time. Once participation
has saturated, the only thing left that moves the price closer to the truth is
more independent information.

## Why pre-IPO onchain markets?

A synthetic onchain market can exist before a public stock market provides an
external anchor. That makes it a useful setting for studying the difference
between market participation and independent information, because for a while
there is nothing else pinning the price down: no exchange close, no index, no
regulated disclosure cadence. The market's information capacity is whatever its
participants can independently learn, and the price is the only visible output.

The repository ships three parameter regimes that illustrate the range — an
*Informative* market with many weakly correlated sources, a *Selected Clientele*
market with fewer correlated sources and a participation tilt, and a *Fast
Follower* market with 2,500 traders reading just two highly correlated sources.
All three can print the same price and mean very different things:

| World | `K` | `rho_s` | SD of `V` given the same price band |
|---|---:|---:|---:|
| Informative | 50 | 0.1 | 0.175 |
| Selected Clientele | 10 | 0.5 | 0.491 |
| Fast Follower | 2 | 0.9 | 0.699 |

**No historical IPO was used to calibrate, classify or validate any of this.**
The worlds are parameter regimes, not companies. See
[research/CLAIMS.md](research/CLAIMS.md) for the full boundary between what this
repository supports and what it does not.

## Quick start

```bash
cargo run --release -- validate       # research-integrity gate, ~0.3 s
cargo run --release -- publication    # every canonical result file, ~1 s
python3 scripts/plot.py               # figures, optional
```

Or reproduce everything from a clean checkout:

```bash
./scripts/reproduce.sh
```

## Commands

| Command | What it produces |
|---|---|
| `validate` | Runs every validation gate; exits non-zero on failure. `results/validation.json` |
| `asymptote` | Simulated MSE against the analytic floor. `results/finite_source_asymptote.csv` |
| `traders-vs-sources` | The `N_T` × `K` grid, the doubling comparison, convergence thresholds |
| `same-price` | Conditional distribution of `V` given the same price band |
| `worlds` | The three regimes through the official-information boundary |
| `sensitivity` | One-factor sweeps and the phase slices |
| `publication` | All of the above, plus `results/summary.json` |

Global flags: `--config <PATH>`, `--seed <SEED>`, `--output-dir <DIR>`,
`--quick`, `--verbose`.

Experiments are configured in TOML, not in flags:

```bash
cargo run --release -- worlds --config configs/publication.toml
```

`configs/publication.toml` is the canonical configuration and states every
parameter of every experiment. `configs/quick.toml` runs the same experiments at
reduced Monte Carlo scale. Both are checked against the in-code defaults by
`tests/config_contract.rs`, so they cannot drift.

## What is in here

```
src/model/        the information hierarchy, one module per layer
src/analytics/    closed-form results and streaming statistics
src/simulation/   the deterministic Monte Carlo engine and the experiments
src/validation/   the research-integrity gate
src/output/       CSV and JSON serialisation
configs/          the canonical experiment definitions
research/         model, validation, claims and Manus parity documentation
scripts/plot.py   the only Python: figures from Rust-generated CSVs
```

Everything computational is Rust: configuration, random number generation,
every layer of the model, market clearing, Monte Carlo, online statistics, the
analytic floor, all experiments, all validation, and all output. Python renders
figures from files Rust has already written. It runs no simulation and computes
no canonical statistic.

## Reproducibility

The master seed is `20260915`. Every stream descends from it through a pure hash
of the work unit's coordinates — experiment, parameter cell, seed block, batch —
so the same configuration and seed reproduce the same numbers regardless of how
many threads ran. `tests/reproducibility.rs` and the `deterministic
reproduction` gate both enforce this. See [research/MODEL.md](research/MODEL.md)
for the seed derivation scheme.

## Reading further

* [research/MODEL.md](research/MODEL.md) — the hierarchy, its assumptions and
  its equations.
* [research/VALIDATION.md](research/VALIDATION.md) — every gate and what it
  would catch.
* [research/CLAIMS.md](research/CLAIMS.md) — supported and unsupported claims.
* [research/MANUS_PARITY.md](research/MANUS_PARITY.md) — agreement with the
  Python research prototype this implementation replaces.

## Scope

Version 0.1 does one thing: simulate and validate how finite independent
information constrains market price informativeness even as trader participation
grows. There are no order books, AMMs, latency, MEV, gas, fees, arbitrage,
liquidation, welfare or capital dynamics, and none of them are needed for the
result.

## License

MIT. See [LICENSE](LICENSE).
