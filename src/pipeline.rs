//! Command orchestration: run experiments, write result files, summarise.
//!
//! The CLI is a thin shell over this module, so every command is callable from
//! a test or another program without going through `main`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Serialize;
use serde_json::json;

use crate::config::Config;
use crate::error::Result;
use crate::output::{write_json, write_rows};
use crate::simulation::{asymptote, same_price, sensitivity, traders_vs_sources, worlds};
use crate::validation::{GateReport, run_gate};

/// File names of the canonical result files.
pub mod files {
    /// Finite-source asymptote benchmark.
    pub const ASYMPTOTE: &str = "finite_source_asymptote.csv";
    /// Traders-versus-sources grid.
    pub const GRID: &str = "traders_vs_sources.csv";
    /// Doubling comparison.
    pub const DOUBLING: &str = "trader_source_doubling.csv";
    /// Convergence thresholds.
    pub const CONVERGENCE: &str = "convergence_thresholds.csv";
    /// Three worlds.
    pub const WORLDS: &str = "worlds.csv";
    /// Same-price conditional distributions.
    pub const SAME_PRICE: &str = "same_price.csv";
    /// One-factor sensitivity sweeps.
    pub const SENSITIVITY: &str = "sensitivity.csv";
    /// Information-independence phase slices.
    pub const PHASE: &str = "information_phase_slices.csv";
    /// Validation report.
    pub const VALIDATION: &str = "validation.json";
    /// Publication run summary.
    pub const SUMMARY: &str = "summary.json";
}

/// Provenance and headline numbers for one command invocation.
#[derive(Debug, Clone, Serialize)]
pub struct RunSummary {
    /// Command that produced the run.
    pub command: String,
    /// Master seed.
    pub master_seed: u64,
    /// Wall-clock seconds.
    pub elapsed_seconds: f64,
    /// Result files written, relative to the output directory.
    pub outputs: Vec<String>,
    /// Headline numbers, for a quick read without opening the CSVs.
    pub headline: serde_json::Value,
    /// Validation report, when the command ran the gate.
    pub gate: Option<GateReport>,
}

fn path(out_dir: &Path, name: &str) -> PathBuf {
    out_dir.join(name)
}

/// Run the finite-source asymptote benchmark and write its CSV.
pub fn run_asymptote(cfg: &Config, out_dir: &Path) -> Result<RunSummary> {
    let start = Instant::now();
    let rows = asymptote::run(cfg)?;
    write_rows(path(out_dir, files::ASYMPTOTE), &rows)?;
    let worst = rows.iter().map(|r| r.z_gap.abs()).fold(0.0f64, f64::max);
    Ok(RunSummary {
        command: "asymptote".into(),
        master_seed: cfg.master_seed,
        elapsed_seconds: start.elapsed().as_secs_f64(),
        outputs: vec![files::ASYMPTOTE.into()],
        headline: json!({
            "cells": rows.len(),
            "realizations_per_cell": cfg.asymptote.realizations,
            "max_abs_z_gap_to_floor": worst,
        }),
        gate: None,
    })
}

/// Run the grid, doubling comparison and convergence thresholds.
pub fn run_traders_vs_sources(cfg: &Config, out_dir: &Path) -> Result<RunSummary> {
    let start = Instant::now();
    let result = traders_vs_sources::run(cfg)?;
    write_rows(path(out_dir, files::GRID), &result.grid)?;
    write_rows(path(out_dir, files::DOUBLING), &result.doubling)?;
    write_rows(path(out_dir, files::CONVERGENCE), &result.convergence)?;
    let doubling: Vec<_> = result
        .doubling
        .iter()
        .map(|row| {
            json!({
                "density": row.density,
                "n_traders": row.n_traders,
                "improvement_double_traders": row.improvement_double_traders,
                "improvement_double_sources": row.improvement_double_sources,
            })
        })
        .collect();
    Ok(RunSummary {
        command: "traders-vs-sources".into(),
        master_seed: cfg.master_seed,
        elapsed_seconds: start.elapsed().as_secs_f64(),
        outputs: vec![
            files::GRID.into(),
            files::DOUBLING.into(),
            files::CONVERGENCE.into(),
        ],
        headline: json!({
            "grid_cells": result.grid.len(),
            "doubling": doubling,
        }),
        gate: None,
    })
}

/// Run the three-worlds experiment.
pub fn run_worlds(cfg: &Config, out_dir: &Path) -> Result<RunSummary> {
    let start = Instant::now();
    let rows = worlds::run(cfg)?;
    write_rows(path(out_dir, files::WORLDS), &rows)?;
    let summary: Vec<_> = rows
        .iter()
        .map(|row| {
            json!({
                "world": row.world,
                "pre_price_mse": row.pre_price_mse,
                "post_price_mse": row.post_price_mse,
                "mean_abs_revision": row.mean_abs_revision,
            })
        })
        .collect();
    Ok(RunSummary {
        command: "worlds".into(),
        master_seed: cfg.master_seed,
        elapsed_seconds: start.elapsed().as_secs_f64(),
        outputs: vec![files::WORLDS.into()],
        headline: json!({ "worlds": summary }),
        gate: None,
    })
}

/// Run the same-price conditional experiment.
pub fn run_same_price(cfg: &Config, out_dir: &Path) -> Result<RunSummary> {
    let start = Instant::now();
    let rows = same_price::run(cfg)?;
    write_rows(path(out_dir, files::SAME_PRICE), &rows)?;
    let summary: Vec<_> = rows
        .iter()
        .map(|row| {
            json!({
                "world": row.world,
                "accepted_n": row.accepted_n,
                "mean_v": row.mean_v,
                "sd_v": row.sd_v,
                "q05_v": row.q05_v,
                "q95_v": row.q95_v,
            })
        })
        .collect();
    Ok(RunSummary {
        command: "same-price".into(),
        master_seed: cfg.master_seed,
        elapsed_seconds: start.elapsed().as_secs_f64(),
        outputs: vec![files::SAME_PRICE.into()],
        headline: json!({ "worlds": summary }),
        gate: None,
    })
}

/// Run the sensitivity sweeps.
pub fn run_sensitivity(cfg: &Config, out_dir: &Path) -> Result<RunSummary> {
    let start = Instant::now();
    let result = sensitivity::run(cfg)?;
    write_rows(path(out_dir, files::SENSITIVITY), &result.sweeps)?;
    write_rows(path(out_dir, files::PHASE), &result.phase)?;
    Ok(RunSummary {
        command: "sensitivity".into(),
        master_seed: cfg.master_seed,
        elapsed_seconds: start.elapsed().as_secs_f64(),
        outputs: vec![files::SENSITIVITY.into(), files::PHASE.into()],
        headline: json!({
            "sweep_points": result.sweeps.len(),
            "phase_cells": result.phase.len(),
        }),
        gate: None,
    })
}

/// Run the validation gate and write its report.
pub fn run_validation(cfg: &Config, out_dir: &Path) -> Result<RunSummary> {
    let start = Instant::now();
    let report = run_gate(cfg)?;
    write_json(path(out_dir, files::VALIDATION), &report)?;
    let passed = report.passed;
    Ok(RunSummary {
        command: "validate".into(),
        master_seed: cfg.master_seed,
        elapsed_seconds: start.elapsed().as_secs_f64(),
        outputs: vec![files::VALIDATION.into()],
        headline: json!({ "passed": passed }),
        gate: Some(report),
    })
}

/// Regenerate every canonical result file required by the technical article.
pub fn run_publication(cfg: &Config, out_dir: &Path) -> Result<RunSummary> {
    let start = Instant::now();
    let mut stages = Vec::new();
    let mut outputs = Vec::new();

    for stage in [
        run_asymptote(cfg, out_dir)?,
        run_traders_vs_sources(cfg, out_dir)?,
        run_worlds(cfg, out_dir)?,
        run_same_price(cfg, out_dir)?,
        run_sensitivity(cfg, out_dir)?,
        run_validation(cfg, out_dir)?,
    ] {
        outputs.extend(stage.outputs.iter().cloned());
        stages.push(stage);
    }

    let gate = stages.iter().rev().find_map(|s| s.gate.clone());
    let headline = json!({
        "stages": stages
            .iter()
            .map(|s| json!({
                "command": s.command,
                "elapsed_seconds": s.elapsed_seconds,
                "headline": s.headline,
            }))
            .collect::<Vec<_>>(),
        "gate_passed": gate.as_ref().map(|g| g.passed),
    });

    let summary = RunSummary {
        command: "publication".into(),
        master_seed: cfg.master_seed,
        elapsed_seconds: start.elapsed().as_secs_f64(),
        outputs: {
            outputs.push(files::SUMMARY.into());
            outputs
        },
        headline,
        gate,
    };
    write_json(path(out_dir, files::SUMMARY), &summary)?;
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publication_writes_every_canonical_file() {
        let mut cfg = Config::quick();
        // Keep the integration cheap: the point is the file contract.
        cfg.asymptote.realizations = 2_000;
        cfg.asymptote.k_values = vec![10];
        cfg.asymptote.rho_values = vec![0.5];
        cfg.asymptote.trader_values = vec![1000];
        cfg.grid.realizations_per_block = 200;
        cfg.grid.trader_values = vec![50, 500];
        cfg.grid.k_values = vec![2, 10];
        cfg.grid.convergence_k_values = vec![10];
        cfg.doubling.realizations = 2_000;
        cfg.worlds.realizations = 2_000;
        cfg.same_price.realizations = 20_000;
        cfg.sensitivity.realizations_per_block = 500;
        cfg.sensitivity.sweeps = crate::config::SweepSpec {
            sigma_s: vec![0.75],
            rho_s: vec![0.5],
            sources: vec![10],
            interpretation_sigma: vec![0.8],
            traders: vec![2500],
            clientele_bias: vec![0.35],
        };
        cfg.sensitivity.phase.slices = vec![crate::config::PhaseSliceSpec {
            label: "moderate".into(),
            sources: 10,
            rho_s: 0.5,
        }];
        cfg.sensitivity.phase.sigma_s_steps = 2;
        cfg.sensitivity.phase.bias_steps = 2;
        cfg.sensitivity.phase.realizations_per_block = 200;
        cfg.validation.realizations = 5_000;
        cfg.validation.same_price_realizations = 20_000;
        cfg.validation.plateau_trader_values = vec![100, 1_000];

        let dir = std::env::temp_dir().join("mic-publication-contract");
        let _ = std::fs::remove_dir_all(&dir);
        let summary = run_publication(&cfg, &dir).expect("publication runs");
        for name in [
            files::ASYMPTOTE,
            files::GRID,
            files::DOUBLING,
            files::CONVERGENCE,
            files::WORLDS,
            files::SAME_PRICE,
            files::SENSITIVITY,
            files::PHASE,
            files::VALIDATION,
            files::SUMMARY,
        ] {
            assert!(dir.join(name).exists(), "missing {name}");
        }
        assert_eq!(summary.command, "publication");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
