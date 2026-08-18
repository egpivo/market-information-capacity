//! End-to-end tests of the binary: exit codes, result files, gate output.

use std::path::Path;
use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_market-information-capacity");

/// A configuration small enough for a test but still economically separated.
const TINY_CONFIG: &str = r#"
master_seed = 20260915
batches = 4

[asymptote]
k_values = [10]
rho_values = [0.5]
trader_values = [1000]
realizations = 5000

[grid]
trader_values = [50, 500]
k_values = [2, 10]
convergence_k_values = [10]
realizations_per_block = 500

[doubling]
realizations = 5000

[worlds]
realizations = 5000

[same_price]
realizations = 40000

[sensitivity]
realizations_per_block = 1000

[sensitivity.phase]
realizations_per_block = 500
sigma_s_steps = 2
bias_steps = 2

[validation]
realizations = 20000
same_price_realizations = 60000
plateau_trader_values = [100, 1000, 10000]
"#;

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("mic-cli-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

fn write_config(dir: &Path) -> std::path::PathBuf {
    let path = dir.join("tiny.toml");
    std::fs::write(&path, TINY_CONFIG).expect("write config");
    path
}

#[test]
fn validate_passes_and_exits_zero() {
    let dir = scratch("validate");
    let config = write_config(&dir);
    let output = Command::new(BIN)
        .args(["validate", "--config"])
        .arg(&config)
        .arg("--output-dir")
        .arg(&dir)
        .output()
        .expect("run validate");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Market Information Capacity — validation"),
        "unexpected output:\n{stdout}"
    );
    assert!(
        stdout.contains("MODEL_GATE: PASS"),
        "gate did not pass:\n{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.status.success(), "validate exited non-zero");
    assert!(dir.join("validation.json").exists());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn validate_exits_non_zero_when_a_gate_fails() {
    // A world configuration that destroys the same-price ordering: give every
    // world the same information environment, so the conditional spreads
    // coincide and the ordering gate must fail.
    let dir = scratch("validate-fail");
    let config = dir.join("broken.toml");
    let mut text = TINY_CONFIG.to_string();
    text.push_str(
        r#"
[[worlds.presets]]
name = "A"
traders = 2500
sources = 10
rho_s = 0.5
sigma_s = 0.75
clientele_bias = 0.35
interpretation_sigma = 0.8

[[worlds.presets]]
name = "B"
traders = 2500
sources = 10
rho_s = 0.5
sigma_s = 0.75
clientele_bias = 0.35
interpretation_sigma = 0.8
"#,
    );
    std::fs::write(&config, text).expect("write config");

    let output = Command::new(BIN)
        .args(["validate", "--config"])
        .arg(&config)
        .arg("--output-dir")
        .arg(&dir)
        .output()
        .expect("run validate");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("MODEL_GATE: FAIL"), "unexpected:\n{stdout}");
    assert!(!output.status.success(), "a failed gate must exit non-zero");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn publication_writes_every_canonical_result_file() {
    let dir = scratch("publication");
    let config = write_config(&dir);
    let status = Command::new(BIN)
        .args(["publication", "--config"])
        .arg(&config)
        .arg("--output-dir")
        .arg(&dir)
        .status()
        .expect("run publication");
    assert!(status.success());

    for name in [
        "finite_source_asymptote.csv",
        "traders_vs_sources.csv",
        "trader_source_doubling.csv",
        "convergence_thresholds.csv",
        "worlds.csv",
        "same_price.csv",
        "sensitivity.csv",
        "information_phase_slices.csv",
        "validation.json",
        "summary.json",
    ] {
        let path = dir.join(name);
        assert!(path.exists(), "missing {name}");
        let text = std::fs::read_to_string(&path).expect("read result");
        assert!(text.lines().count() > 1, "{name} has no rows");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn seed_override_changes_results_but_not_the_file_contract() {
    let dir = scratch("seed");
    let config = write_config(&dir);
    let run = |seed: &str, out: &Path| {
        let status = Command::new(BIN)
            .args(["worlds", "--config"])
            .arg(&config)
            .arg("--seed")
            .arg(seed)
            .arg("--output-dir")
            .arg(out)
            .status()
            .expect("run worlds");
        assert!(status.success());
        std::fs::read_to_string(out.join("worlds.csv")).expect("read worlds.csv")
    };
    let a = run("20260915", &dir.join("a"));
    let b = run("20260916", &dir.join("b"));
    assert_ne!(a, b, "changing the seed must change the results");
    assert_eq!(
        a.lines().next(),
        b.lines().next(),
        "the CSV header must not depend on the seed"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_bad_config_is_rejected_with_a_message() {
    let dir = scratch("bad-config");
    let config = dir.join("bad.toml");
    std::fs::write(&config, "[[worlds.presets]]\nname = \"X\"\ntraders = 10\nsources = 2\nrho_s = 1.5\nsigma_s = 1.0\nclientele_bias = 0.0\ninterpretation_sigma = 0.8\n")
        .expect("write config");
    let output = Command::new(BIN)
        .args(["worlds", "--config"])
        .arg(&config)
        .arg("--output-dir")
        .arg(&dir)
        .output()
        .expect("run worlds");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("rho_s") || stderr.contains("correlation"),
        "unhelpful error:\n{stderr}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
