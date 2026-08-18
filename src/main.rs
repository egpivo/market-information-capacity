//! Command-line interface for the market-information-capacity research tool.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Context;
use clap::{Parser, Subcommand};
use market_information_capacity::config::Config;
use market_information_capacity::pipeline;

/// Simulate and validate how finite independent information constrains market
/// price informativeness as trader participation grows.
#[derive(Debug, Parser)]
#[command(name = "market-information-capacity", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,

    /// TOML configuration file. Omitted sections fall back to canonical values.
    #[arg(long, global = true, value_name = "PATH")]
    config: Option<PathBuf>,

    /// Override the master seed.
    #[arg(long, global = true, value_name = "SEED")]
    seed: Option<u64>,

    /// Directory for result files.
    #[arg(long, global = true, value_name = "DIR", default_value = "results")]
    output_dir: PathBuf,

    /// Use the reduced-scale configuration instead of publication scale.
    #[arg(long, global = true)]
    quick: bool,

    /// Print the evidence behind every validation check.
    #[arg(long, global = true)]
    verbose: bool,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run the research-integrity gate; exits non-zero if any check fails.
    Validate,
    /// Benchmark simulated price MSE against the analytic finite-source floor.
    Asymptote,
    /// Sweep trader count against source budget, with the doubling comparison.
    TradersVsSources,
    /// Conditional distribution of the latent value given the same price.
    SamePrice,
    /// The three canonical parameter regimes through the official boundary.
    Worlds,
    /// One-factor and phase sensitivity sweeps.
    Sensitivity,
    /// Regenerate every canonical result file.
    Publication,
}

fn load_config(cli: &Cli) -> anyhow::Result<Config> {
    let mut cfg = match &cli.config {
        Some(path) => Config::load(path).with_context(|| format!("loading {}", path.display()))?,
        None if cli.quick => Config::quick(),
        None => Config::publication(),
    };
    if let Some(seed) = cli.seed {
        cfg.master_seed = seed;
    }
    cfg.validate().context("validating configuration")?;
    Ok(cfg)
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .without_time()
        .init();

    match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> anyhow::Result<ExitCode> {
    let cli = Cli::parse();
    let cfg = load_config(&cli)?;
    let out_dir = cli.output_dir.as_path();

    let summary = match cli.command {
        Command::Validate => pipeline::run_validation(&cfg, out_dir)?,
        Command::Asymptote => pipeline::run_asymptote(&cfg, out_dir)?,
        Command::TradersVsSources => pipeline::run_traders_vs_sources(&cfg, out_dir)?,
        Command::SamePrice => pipeline::run_same_price(&cfg, out_dir)?,
        Command::Worlds => pipeline::run_worlds(&cfg, out_dir)?,
        Command::Sensitivity => pipeline::run_sensitivity(&cfg, out_dir)?,
        Command::Publication => pipeline::run_publication(&cfg, out_dir)?,
    };

    if let Some(gate) = &summary.gate {
        print!("{}", gate.render());
        if cli.verbose {
            println!("\nevidence:\n{}", gate.render_evidence());
        }
    }

    tracing::info!(
        command = %summary.command,
        seed = summary.master_seed,
        elapsed_s = format!("{:.1}", summary.elapsed_seconds),
        outputs = summary.outputs.join(", "),
        "run complete"
    );

    let failed = summary.gate.as_ref().is_some_and(|gate| !gate.passed);
    Ok(if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    })
}
