//! Sensitivity of the information-capacity results to the model's parameters.
//!
//! Two complementary sweeps:
//!
//! * one-factor-at-a-time perturbations of a baseline regime along the six
//!   dimensions that define the information environment — source precision,
//!   source correlation, source count, interpretation noise, trader count and
//!   clientele tilt;
//! * a two-dimensional phase sweep of source precision against clientele tilt
//!   inside three fixed information regimes, reported as the relative gap to
//!   the analytic floor on a common scale.
//!
//! The output is descriptive. This repository studies information capacity
//! only: no welfare, overtrading or liquidity quantity is computed or claimed.

use serde::Serialize;

use crate::analytics::information_floor::{analytic_floor, analytic_pre_price_mse};
use crate::config::{Config, PhaseSliceSpec, WorldSpec};
use crate::error::Result;
use crate::simulation::monte_carlo::{Cell, block_summary, run_world};

/// Stable experiment label for the one-factor sweeps.
pub const SWEEP_EXPERIMENT: &str = "sensitivity-sweep";
/// Stable experiment label for the phase sweep.
pub const PHASE_EXPERIMENT: &str = "sensitivity-phase";

/// One point of a one-factor sweep.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SensitivityRow {
    /// Which parameter was varied.
    pub dimension: String,
    /// The value it took.
    pub value: f64,
    /// Trader count at this point.
    pub n_traders: usize,
    /// Source budget at this point.
    pub k_sources: usize,
    /// Source correlation at this point.
    pub rho_s: f64,
    /// Source error scale at this point.
    pub sigma_s: f64,
    /// Clientele tilt at this point.
    pub clientele_bias: f64,
    /// Interpretation noise at this point.
    pub interpretation_sigma: f64,
    /// Analytic asymptotic floor.
    pub analytic_floor: f64,
    /// Simulated pre-official price MSE.
    pub pre_price_mse: f64,
    /// Standard error across seed blocks.
    pub block_se: f64,
    /// Exact pre-official price MSE.
    pub analytic_pre_price_mse: f64,
    /// Relative distance of the simulated MSE above the floor.
    pub gap_to_floor: f64,
    /// Simulated post-boundary price MSE.
    pub post_price_mse: f64,
    /// Simulated mean absolute revision.
    pub mean_abs_revision: f64,
}

/// One cell of the phase sweep.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PhaseRow {
    /// Slice label.
    pub slice: String,
    /// Source budget of the slice.
    pub k_sources: usize,
    /// Source correlation of the slice.
    pub rho_s: f64,
    /// Source error scale at this cell.
    pub sigma_s: f64,
    /// Clientele tilt at this cell.
    pub clientele_bias: f64,
    /// Trader count held fixed.
    pub n_traders: usize,
    /// Simulated price MSE.
    pub mse_price: f64,
    /// Standard error across seed blocks.
    pub block_se: f64,
    /// Analytic asymptotic floor.
    pub analytic_floor: f64,
    /// Relative gap to the floor.
    pub gap_to_floor: f64,
}

/// Both sensitivity outputs.
#[derive(Debug, Clone)]
pub struct Sensitivity {
    /// One-factor sweeps.
    pub sweeps: Vec<SensitivityRow>,
    /// Phase sweep.
    pub phase: Vec<PhaseRow>,
}

/// Blocked run of one regime: pre MSE with block standard error, plus post MSE
/// and mean absolute revision.
struct RegimeResult {
    pre_mse: f64,
    block_se: f64,
    post_mse: f64,
    mean_abs_revision: f64,
}

fn run_regime(
    cfg: &Config,
    experiment: &str,
    cell: u64,
    blocks: usize,
    schedule: &[usize],
    world: &WorldSpec,
) -> Result<RegimeResult> {
    let market = world.market(cfg.official);
    let mut pre = Vec::with_capacity(blocks);
    let mut post = Vec::with_capacity(blocks);
    let mut revision = Vec::with_capacity(blocks);
    for block in 0..blocks {
        // Revision deciles are not reported for sweeps, so no samples are retained.
        let acc = run_world(
            Cell::new(cfg.master_seed, experiment, cell, schedule).with_block(block as u64),
            &market,
            0,
        )?;
        pre.push(acc.pre_mse.mean());
        post.push(acc.post_mse.mean());
        revision.push(acc.abs_revision.mean());
    }
    let (pre_mse, block_se) = block_summary(&pre);
    let (post_mse, _) = block_summary(&post);
    let (mean_abs_revision, _) = block_summary(&revision);
    Ok(RegimeResult {
        pre_mse,
        block_se,
        post_mse,
        mean_abs_revision,
    })
}

fn sweep_row(
    cfg: &Config,
    cell: &mut u64,
    schedule: &[usize],
    dimension: &str,
    value: f64,
    world: WorldSpec,
) -> Result<SensitivityRow> {
    let result = run_regime(
        cfg,
        SWEEP_EXPERIMENT,
        *cell,
        cfg.sensitivity.blocks,
        schedule,
        &world,
    )?;
    *cell += 1;
    let floor = analytic_floor(world.sources, world.rho_s, world.sigma_s);
    Ok(SensitivityRow {
        dimension: dimension.to_string(),
        value,
        n_traders: world.traders,
        k_sources: world.sources,
        rho_s: world.rho_s,
        sigma_s: world.sigma_s,
        clientele_bias: world.clientele_bias,
        interpretation_sigma: world.interpretation_sigma,
        analytic_floor: floor,
        pre_price_mse: result.pre_mse,
        block_se: result.block_se,
        analytic_pre_price_mse: analytic_pre_price_mse(&world.market(cfg.official)),
        gap_to_floor: (result.pre_mse - floor) / floor,
        post_price_mse: result.post_mse,
        mean_abs_revision: result.mean_abs_revision,
    })
}

/// Run the one-factor sweeps.
pub fn run_sweeps(cfg: &Config) -> Result<Vec<SensitivityRow>> {
    let schedule = cfg.batch_schedule(cfg.sensitivity.realizations_per_block);
    let baseline = &cfg.sensitivity.baseline;
    let sweeps = &cfg.sensitivity.sweeps;
    let mut rows = Vec::new();
    let mut cell: u64 = 0;

    for &value in &sweeps.sigma_s {
        let mut world = baseline.clone();
        world.sigma_s = value;
        rows.push(sweep_row(
            cfg,
            &mut cell,
            &schedule,
            "source_precision",
            value,
            world,
        )?);
    }
    for &value in &sweeps.rho_s {
        let mut world = baseline.clone();
        world.rho_s = value;
        rows.push(sweep_row(
            cfg,
            &mut cell,
            &schedule,
            "source_correlation",
            value,
            world,
        )?);
    }
    for &value in &sweeps.sources {
        let mut world = baseline.clone();
        world.sources = value;
        rows.push(sweep_row(
            cfg,
            &mut cell,
            &schedule,
            "source_count",
            value as f64,
            world,
        )?);
    }
    for &value in &sweeps.interpretation_sigma {
        let mut world = baseline.clone();
        world.interpretation_sigma = value;
        rows.push(sweep_row(
            cfg,
            &mut cell,
            &schedule,
            "interpretation_noise",
            value,
            world,
        )?);
    }
    for &value in &sweeps.traders {
        let mut world = baseline.clone();
        world.traders = value;
        rows.push(sweep_row(
            cfg,
            &mut cell,
            &schedule,
            "trader_count",
            value as f64,
            world,
        )?);
    }
    for &value in &sweeps.clientele_bias {
        let mut world = baseline.clone();
        world.clientele_bias = value;
        rows.push(sweep_row(
            cfg,
            &mut cell,
            &schedule,
            "clientele_bias",
            value,
            world,
        )?);
    }
    Ok(rows)
}

/// Geometric grid of `steps` points from `min` to `max`.
fn geomspace(min: f64, max: f64, steps: usize) -> Vec<f64> {
    if steps <= 1 {
        return vec![min];
    }
    let (lo, hi) = (min.ln(), max.ln());
    (0..steps)
        .map(|i| (lo + (hi - lo) * i as f64 / (steps - 1) as f64).exp())
        .collect()
}

/// Linear grid of `steps` points from `min` to `max`.
fn linspace(min: f64, max: f64, steps: usize) -> Vec<f64> {
    if steps <= 1 {
        return vec![min];
    }
    (0..steps)
        .map(|i| min + (max - min) * i as f64 / (steps - 1) as f64)
        .collect()
}

fn phase_world(slice: &PhaseSliceSpec, cfg: &Config, sigma_s: f64, bias: f64) -> WorldSpec {
    WorldSpec {
        name: slice.label.clone(),
        traders: cfg.sensitivity.phase.traders,
        sources: slice.sources,
        rho_s: slice.rho_s,
        sigma_s,
        clientele_bias: bias,
        interpretation_sigma: cfg.sensitivity.phase.interpretation_sigma,
    }
}

/// Run the phase sweep.
pub fn run_phase(cfg: &Config) -> Result<Vec<PhaseRow>> {
    let phase = &cfg.sensitivity.phase;
    let schedule = cfg.batch_schedule(phase.realizations_per_block);
    let sigmas = geomspace(phase.sigma_s_min, phase.sigma_s_max, phase.sigma_s_steps);
    let biases = linspace(phase.bias_min, phase.bias_max, phase.bias_steps);
    let mut rows = Vec::new();
    let mut cell: u64 = 0;
    for slice in &phase.slices {
        for &sigma_s in &sigmas {
            for &bias in &biases {
                let world = phase_world(slice, cfg, sigma_s, bias);
                let result =
                    run_regime(cfg, PHASE_EXPERIMENT, cell, phase.blocks, &schedule, &world)?;
                cell += 1;
                let floor = analytic_floor(slice.sources, slice.rho_s, sigma_s);
                rows.push(PhaseRow {
                    slice: slice.label.clone(),
                    k_sources: slice.sources,
                    rho_s: slice.rho_s,
                    sigma_s,
                    clientele_bias: bias,
                    n_traders: phase.traders,
                    mse_price: result.pre_mse,
                    block_se: result.block_se,
                    analytic_floor: floor,
                    gap_to_floor: (result.pre_mse - floor) / floor,
                });
            }
        }
    }
    Ok(rows)
}

/// Run both sensitivity sweeps.
pub fn run(cfg: &Config) -> Result<Sensitivity> {
    Ok(Sensitivity {
        sweeps: run_sweeps(cfg)?,
        phase: run_phase(cfg)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grids_are_well_formed() {
        let g = geomspace(0.35, 1.8, 9);
        assert_eq!(g.len(), 9);
        assert!((g[0] - 0.35).abs() < 1e-12);
        assert!((g[8] - 1.8).abs() < 1e-12);
        assert!(g.windows(2).all(|w| w[1] > w[0]));

        let l = linspace(0.0, 1.5, 9);
        assert_eq!(l.len(), 9);
        assert!((l[8] - 1.5).abs() < 1e-12);
    }

    #[test]
    fn source_count_sweep_lowers_the_floor_monotonically() {
        let mut cfg = Config::quick();
        cfg.sensitivity.realizations_per_block = 2_000;
        cfg.sensitivity.sweeps.sigma_s.clear();
        cfg.sensitivity.sweeps.rho_s.clear();
        cfg.sensitivity.sweeps.interpretation_sigma.clear();
        cfg.sensitivity.sweeps.traders.clear();
        cfg.sensitivity.sweeps.clientele_bias.clear();
        let rows = run_sweeps(&cfg).expect("sweeps run");
        assert!(rows.iter().all(|r| r.dimension == "source_count"));
        assert!(
            rows.windows(2)
                .all(|w| w[1].analytic_floor <= w[0].analytic_floor),
            "floor must fall as the source budget grows"
        );
    }
}
