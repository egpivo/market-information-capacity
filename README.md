# Market Information Capacity

Deterministic Rust simulation toolkit for studying how **finite independent information constrains market price informativeness as trader participation grows**.

The model separates:

* $N_T$ — number of traders
* $K$ — number of independent information sources

More traders do not mechanically create more information. With a fixed source budget, price error approaches

$$
\sigma_s^2\left[\rho_s+\frac{1-\rho_s}{K}\right]+b^2,
$$

which does not depend on $N_T$.

This repository contains simulation experiments only: no market-data calibration, welfare analysis, or optimal-participation claim.

## Requirements

* Rust 1.85+
* Cargo
* Python 3 with `pandas` and `matplotlib` for figures

## Run

```bash
git clone https://github.com/egpivo/market-information-capacity.git
cd market-information-capacity
cargo build --release

cargo test --release
cargo run --release -- validate --verbose
cargo run --release -- publication --config configs/publication.toml
```

Key experiments:

```bash
cargo run --release -- asymptote
cargo run --release -- traders-vs-sources
cargo run --release -- same-price
cargo run --release -- worlds
cargo run --release -- sensitivity
cargo run --release -- scale-participation --config configs/publication.toml
```

Generate figures:

```bash
python3 scripts/plot.py
python3 scripts/make_participation_gif.py
```

Or reproduce the full pipeline:

```bash
./scripts/reproduce.sh
```

## License

MIT
