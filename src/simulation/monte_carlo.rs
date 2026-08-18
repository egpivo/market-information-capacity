//! The deterministic, batched Monte Carlo engine.
//!
//! Every experiment in this crate goes through [`run_batches`]. A block of
//! realizations is split into a fixed batch schedule before any thread starts;
//! each batch owns a stream derived from its coordinates; batch accumulators
//! are merged back in batch-index order. Parallelism therefore changes runtime
//! and nothing else.
//!
//! Accumulators are streaming: no experiment materialises a sample matrix.

use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::analytics::metrics::{OnlineStats, SampleReservoir};
use crate::config::MarketConfig;
use crate::error::Result;
use crate::model::market::MarketSimulator;
use crate::rng::{StreamId, derive_seed, stream};

/// An accumulator that can be combined across parallel batches.
pub trait Merge {
    /// Fold another batch's accumulator into this one.
    fn merge(&mut self, other: &Self);
}

impl Merge for OnlineStats {
    fn merge(&mut self, other: &Self) {
        OnlineStats::merge(self, other);
    }
}

/// Coordinates and size of one Monte Carlo block.
///
/// Bundling these together keeps experiment call sites readable and makes it
/// hard to pass a stream coordinate in the wrong position.
#[derive(Debug, Clone, Copy)]
pub struct Cell<'a> {
    /// Master seed the run descends from.
    pub master_seed: u64,
    /// Stable experiment label.
    pub experiment: &'a str,
    /// Parameter-cell index within the experiment.
    pub index: u64,
    /// Seed-block index within the cell.
    pub block: u64,
    /// Per-batch realization schedule, fixed before any thread starts.
    pub schedule: &'a [usize],
}

impl<'a> Cell<'a> {
    /// Build a block descriptor at block zero.
    pub fn new(master_seed: u64, experiment: &'a str, index: u64, schedule: &'a [usize]) -> Self {
        Self {
            master_seed,
            experiment,
            index,
            block: 0,
            schedule,
        }
    }

    /// The same cell at a different seed block.
    pub fn with_block(self, block: u64) -> Self {
        Self { block, ..self }
    }

    /// Seed of this block's first batch, for auditability.
    pub fn seed(&self) -> u64 {
        derive_seed(
            self.master_seed,
            StreamId::new(self.experiment, self.index, self.block, 0),
        )
    }
}

/// Run a block of realizations as deterministic parallel batches.
///
/// `step` is invoked once per batch with that batch's accumulator, its private
/// stream, and the number of realizations it owns.
pub fn run_batches<A, I, S>(cell: Cell<'_>, init: I, step: S) -> A
where
    A: Merge + Send,
    I: Fn() -> A + Sync,
    S: Fn(&mut A, &mut ChaCha8Rng, usize) + Sync,
{
    let parts: Vec<A> = cell
        .schedule
        .par_iter()
        .enumerate()
        .map(|(batch, &count)| {
            let mut rng = stream(
                cell.master_seed,
                StreamId::new(cell.experiment, cell.index, cell.block, batch as u64),
            );
            let mut acc = init();
            step(&mut acc, &mut rng, count);
            acc
        })
        .collect();

    let mut out = init();
    for part in &parts {
        out.merge(part);
    }
    out
}

/// Monte Carlo estimate of the pre-official price MSE for one market.
pub fn pre_price_mse(cell: Cell<'_>, market: &MarketConfig) -> Result<OnlineStats> {
    let sim = MarketSimulator::new(market)?;
    Ok(run_batches(cell, OnlineStats::new, |acc, rng, count| {
        let mut ws = sim.workspace();
        for _ in 0..count {
            acc.push(sim.realize(&mut ws, rng).squared_error());
        }
    }))
}

/// Everything the three-worlds experiment measures in one pass.
#[derive(Debug, Clone)]
pub struct WorldAccumulator {
    /// Squared pre-official pricing error.
    pub pre_mse: OnlineStats,
    /// Squared post-boundary pricing error.
    pub post_mse: OnlineStats,
    /// Squared error of an uninformed price of zero, the reference point for
    /// how much the market knew before the boundary.
    pub uninformed_mse: OnlineStats,
    /// Absolute price revision across the boundary.
    pub abs_revision: OnlineStats,
    /// Retained absolute revisions, for deciles.
    pub revision_samples: SampleReservoir,
}

impl WorldAccumulator {
    /// A fresh accumulator retaining up to `capacity` revisions.
    pub fn new(capacity: usize) -> Self {
        Self {
            pre_mse: OnlineStats::new(),
            post_mse: OnlineStats::new(),
            uninformed_mse: OnlineStats::new(),
            abs_revision: OnlineStats::new(),
            revision_samples: SampleReservoir::new(capacity),
        }
    }
}

impl Merge for WorldAccumulator {
    fn merge(&mut self, other: &Self) {
        self.pre_mse.merge(&other.pre_mse);
        self.post_mse.merge(&other.post_mse);
        self.uninformed_mse.merge(&other.uninformed_mse);
        self.abs_revision.merge(&other.abs_revision);
        self.revision_samples.merge(&other.revision_samples);
    }
}

/// Run one world through the official-information boundary.
pub fn run_world(
    cell: Cell<'_>,
    market: &MarketConfig,
    revision_samples: usize,
) -> Result<WorldAccumulator> {
    let sim = MarketSimulator::new(market)?;
    let per_batch = revision_samples.div_ceil(cell.schedule.len().max(1));
    Ok(run_batches(
        cell,
        || WorldAccumulator::new(per_batch),
        |acc, rng, count| {
            let mut ws = sim.workspace();
            for _ in 0..count {
                let revised = sim.realize_revised(&mut ws, rng);
                acc.pre_mse.push(revised.pre.squared_error());
                acc.post_mse.push(revised.post_squared_error());
                acc.uninformed_mse
                    .push(revised.pre.latent * revised.pre.latent);
                let revision = revised.abs_revision();
                acc.abs_revision.push(revision);
                acc.revision_samples.push(revision, rng);
            }
        },
    ))
}

/// Latent values retained conditional on a narrow price band.
#[derive(Debug, Clone)]
pub struct ConditionalAccumulator {
    /// Realizations drawn, accepted or not.
    pub drawn: u64,
    /// Accepted latent values.
    pub accepted: SampleReservoir,
}

impl ConditionalAccumulator {
    /// A fresh accumulator retaining up to `capacity` accepted values.
    pub fn new(capacity: usize) -> Self {
        Self {
            drawn: 0,
            accepted: SampleReservoir::new(capacity),
        }
    }
}

impl Merge for ConditionalAccumulator {
    fn merge(&mut self, other: &Self) {
        self.drawn += other.drawn;
        self.accepted.merge(&other.accepted);
    }
}

/// Draw realizations and retain the latent value whenever `|P_OC| < band`.
pub fn run_conditional(
    cell: Cell<'_>,
    market: &MarketConfig,
    band: f64,
    retained: usize,
) -> Result<ConditionalAccumulator> {
    let sim = MarketSimulator::new(market)?;
    let per_batch = retained.div_ceil(cell.schedule.len().max(1));
    Ok(run_batches(
        cell,
        || ConditionalAccumulator::new(per_batch),
        |acc, rng, count| {
            let mut ws = sim.workspace();
            for _ in 0..count {
                let realization = sim.realize(&mut ws, rng);
                acc.drawn += 1;
                if realization.pre_price.abs() < band {
                    acc.accepted.push(realization.latent, rng);
                }
            }
        },
    ))
}

/// Mean and block standard error across independent seed blocks.
///
/// The block mean is the reported estimate; the standard error is the sample
/// standard deviation of block means divided by the square root of the block
/// count, matching the prototype's block design.
pub fn block_summary(block_means: &[f64]) -> (f64, f64) {
    let n = block_means.len();
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let mean = block_means.iter().sum::<f64>() / n as f64;
    if n < 2 {
        return (mean, f64::NAN);
    }
    let var = block_means.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    (mean, (var / n as f64).sqrt())
}

/// A blocked Monte Carlo estimate.
///
/// Two standard errors are reported because they answer different questions.
/// `block_se` is the spread of independent seed-block means, which is what the
/// Python prototype reported but which carries only `blocks - 1` degrees of
/// freedom. `pooled` carries every realization, so `pooled.std_error()` is the
/// standard error to use when judging whether a cell sits on its analytic
/// floor.
#[derive(Debug, Clone, Copy)]
pub struct BlockedMse {
    /// Mean of the block means, the reported estimate.
    pub mean: f64,
    /// Standard error across seed blocks.
    pub block_se: f64,
    /// Realization-level accumulator pooled across every block.
    pub pooled: OnlineStats,
}

/// The mean squared pre-official error over several independent seed blocks.
pub fn blocked_pre_price_mse(
    cell: Cell<'_>,
    blocks: usize,
    market: &MarketConfig,
) -> Result<BlockedMse> {
    let mut block_means = Vec::with_capacity(blocks);
    let mut pooled = OnlineStats::new();
    for block in 0..blocks {
        let stats = pre_price_mse(cell.with_block(block as u64), market)?;
        block_means.push(stats.mean());
        pooled.merge(&stats);
    }
    let (mean, block_se) = block_summary(&block_means);
    Ok(BlockedMse {
        mean,
        block_se,
        pooled,
    })
}

/// A single realization helper used by validation checks that need raw draws.
pub fn sample_prices<R: Rng + ?Sized>(
    sim: &MarketSimulator,
    rng: &mut R,
    n: usize,
) -> Vec<crate::model::market::Realization> {
    let mut ws = sim.workspace();
    (0..n).map(|_| sim.realize(&mut ws, rng)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, SourceConfig, TraderConfig};
    use crate::model::traders::{InterpretationNoise, Representation};

    fn market(k: usize, n: usize) -> MarketConfig {
        MarketConfig {
            sources: SourceConfig {
                count: k,
                sigma: 1.0,
                correlation: 0.5,
            },
            traders: TraderConfig {
                count: n,
                interpretation_sigma: 0.8,
                clientele_bias: 0.0,
                representation: Representation::Equal,
                noise: InterpretationNoise::Aggregated,
            },
            official: Default::default(),
        }
    }

    #[test]
    fn batched_run_is_independent_of_batch_count() {
        // The stream partition changes with the batch count, so results are not
        // bit-identical; they must agree well within Monte Carlo error.
        let cfg = market(10, 1000);
        let single = [40_000usize];
        let a = pre_price_mse(Cell::new(20_260_915, "test-batching", 0, &single), &cfg)
            .expect("run")
            .mean();
        let schedule = crate::config::batch_schedule(40_000, 8);
        let b = pre_price_mse(Cell::new(20_260_915, "test-batching", 0, &schedule), &cfg)
            .expect("run")
            .mean();
        assert!((a - b).abs() < 0.02, "{a} vs {b}");
    }

    #[test]
    fn repeated_runs_are_bit_identical() {
        let cfg = market(5, 500);
        let schedule = Config::quick().batch_schedule(10_000);
        let cell = Cell::new(20_260_915, "test-determinism", 3, &schedule).with_block(1);
        let a = pre_price_mse(cell, &cfg).expect("run").mean();
        let b = pre_price_mse(cell, &cfg).expect("run").mean();
        assert_eq!(a.to_bits(), b.to_bits());
    }

    #[test]
    fn block_summary_reports_spread_of_block_means() {
        let (mean, se) = block_summary(&[1.0, 2.0, 3.0]);
        assert!((mean - 2.0).abs() < 1e-12);
        assert!((se - (1.0f64 / 3.0).sqrt()).abs() < 1e-12);
    }
}
