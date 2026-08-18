#!/usr/bin/env bash
#
# Reproduce every canonical result in this repository from a clean checkout.
#
# The Rust pipeline is self-contained: it needs a stable Rust toolchain and
# nothing else. Figure rendering additionally needs Python 3 with pandas and
# matplotlib; if those are missing the scientific pipeline still completes and
# only the figures are skipped.
#
# Usage:
#   ./scripts/reproduce.sh              # publication scale, ~30 s on 10 cores
#   ./scripts/reproduce.sh --quick      # reduced scale, for a fast smoke test

set -euo pipefail

cd "$(dirname "$0")/.."

SCALE_ARGS=()
if [[ "${1:-}" == "--quick" ]]; then
  SCALE_ARGS=(--config configs/quick.toml)
  echo "==> reduced-scale run (configs/quick.toml)"
else
  SCALE_ARGS=(--config configs/publication.toml)
  echo "==> publication-scale run (configs/publication.toml)"
fi

echo "==> cargo fmt --check"
cargo fmt --check

echo "==> cargo clippy"
cargo clippy --all-targets --all-features -- -D warnings

echo "==> cargo test --release"
cargo test --release

echo "==> validate"
cargo run --release --quiet -- validate "${SCALE_ARGS[@]}" --verbose

echo "==> publication"
cargo run --release --quiet -- publication "${SCALE_ARGS[@]}"

echo "==> figures"
if python3 -c "import pandas, matplotlib" >/dev/null 2>&1; then
  python3 scripts/plot.py
else
  echo "    skipped: pandas and matplotlib are not installed."
  echo "    install them with: python3 -m pip install pandas matplotlib"
fi

echo
echo "==> done. results/ holds every canonical CSV and JSON; figures/ holds the PNGs."
