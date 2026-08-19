//! The participation-scale experiment: one trader to one million.
//!
//! The canonical grid starts at `N_T` = 50, which is already inside the
//! asymptotic region, so it shows three flat lines. This experiment sweeps the
//! trader count across six orders of magnitude with the information budget held
//! fixed, which is the range over which the transition from rapid improvement to
//! saturation is visible.
//!
//! # Why this is cheap
//!
//! Under the canonical aggregated-noise representation the trader count enters a
//! realization only through the variance of a single draw, `sigma_nu^2 / N_T`.
//! A cell at `N_T` = 1,000,000 therefore costs exactly what a cell at `N_T` = 1
//! costs. That is not an approximation — the population mean of `N_T` i.i.d.
//! normals is exactly `N(0, sigma_nu^2 / N_T)` — and
//! [`equivalence`](Self::equivalence) checks it against an explicit
//! per-trader implementation at tractable sizes before the large cells are
//! trusted.
//!
//! The experiment is therefore economically large and computationally small.

use serde::Serialize;

use crate::analytics::information_floor::{
    analytic_distance_to_floor, analytic_doubling_gain, analytic_floor, analytic_pre_price_mse,
    analytic_relative_gap,
};
use crate::config::{Config, MarketConfig, SourceConfig, TraderConfig};
use crate::error::Result;
use crate::model::traders::{InterpretationNoise, Representation};
use crate::simulation::experiment::{Experiment, SimulationContext};
use crate::simulation::monte_carlo::{Cell, pre_price_mse};

/// Stable experiment label used for seed derivation.
pub const EXPERIMENT: &str = "participation-scale";
/// Stable label for the aggregated-versus-explicit equivalence check.
pub const EQUIVALENCE_EXPERIMENT: &str = "participation-scale-equivalence";

/// One point on the participation curve.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ScaleRow {
    /// Source budget `K`.
    pub k_sources: usize,
    /// Trader count `N_T`.
    pub n_traders: usize,
    /// Source correlation.
    pub rho_s: f64,
    /// Source error scale.
    pub sigma_s: f64,
    /// Interpretation noise scale.
    pub interpretation_sigma: f64,
    /// Clientele tilt.
    pub clientele_bias: f64,
    /// Asymptotic information floor.
    pub analytic_floor: f64,
    /// Exact finite-`N_T` price MSE.
    pub analytic_mse: f64,
    /// Absolute distance above the floor.
    pub distance_to_floor: f64,
    /// Distance above the floor as a fraction of the floor.
    pub relative_gap: f64,
    /// Traders per source at this point.
    pub traders_per_source: f64,
    /// Whether the canonical equal-representation assumption is doing work
    /// beyond what a literal assignment could deliver (`N_T < K`).
    ///
    /// Below one trader per source the model still credits the price with the
    /// full `K`-source aggregate. That is the frozen v4 assumption, and it is
    /// flagged here so no reader mistakes the small-`N_T` end of the curve for a
    /// statement about markets with fewer participants than sources.
    pub full_coverage_assumed: bool,
    /// Simulated price MSE, where this point was validated.
    pub mc_mse: Option<f64>,
    /// Standard error of the simulated MSE.
    pub mc_se: Option<f64>,
    /// Realizations behind the simulated MSE.
    pub mc_realizations: Option<u64>,
    /// Deviation of the simulated MSE from the analytic curve, in standard errors.
    pub z_gap: Option<f64>,
}

/// The informational return to doubling participation at one point.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct MarginalGainRow {
    /// Source budget `K`.
    pub k_sources: usize,
    /// Base trader count `N_T`.
    pub n_traders: usize,
    /// Exact MSE at `N_T`.
    pub analytic_mse: f64,
    /// Exact MSE at `2 N_T`.
    pub analytic_mse_doubled: f64,
    /// Exact gain `MSE(N_T) - MSE(2 N_T)`.
    pub analytic_gain: f64,
    /// `gain * N_T`, constant under the model and equal to `sigma_nu^2 / 2`.
    pub analytic_gain_times_n: f64,
    /// Simulated gain, where both arms were validated.
    pub mc_gain: Option<f64>,
    /// Standard error of the simulated gain.
    pub mc_gain_se: Option<f64>,
    /// Whether the simulated gain is separable from zero at three standard errors.
    pub mc_gain_resolved: Option<bool>,
}

/// Outcome of the aggregated-versus-explicit representation check.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct EquivalenceRow {
    /// Trader count.
    pub n_traders: usize,
    /// Source budget.
    pub k_sources: usize,
    /// Whether the trader population can cover every source (`N_T >= K`).
    ///
    /// When it cannot, the two representations are *expected* to differ: the
    /// canonical equal-representation assumption keeps all `K` sources in the
    /// price, while a literal round-robin assignment leaves `K - N_T` of them
    /// unread. Those rows are reported, not gated.
    pub full_source_coverage: bool,
    /// MSE under the aggregated closed-form belief representation.
    pub aggregated_mse: f64,
    /// MSE under the explicit per-trader representation.
    pub explicit_mse: f64,
    /// Exact value both should estimate.
    pub analytic_mse: f64,
    /// Difference in standard errors of the difference.
    pub z_gap: f64,
}

/// Everything the experiment produces.
#[derive(Debug, Clone, Serialize)]
pub struct ParticipationScale {
    /// The participation curve.
    pub curve: Vec<ScaleRow>,
    /// The marginal-gain table.
    pub marginal_gain: Vec<MarginalGainRow>,
    /// The representation-equivalence check.
    pub equivalence: Vec<EquivalenceRow>,
}

fn market(cfg: &Config, k: usize, n_traders: usize, noise: InterpretationNoise) -> MarketConfig {
    let scale = &cfg.participation_scale;
    MarketConfig {
        sources: SourceConfig {
            count: k,
            sigma: scale.sigma_s,
            correlation: scale.rho_s,
        },
        traders: TraderConfig {
            count: n_traders,
            interpretation_sigma: scale.interpretation_sigma,
            clientele_bias: scale.clientele_bias,
            representation: Representation::Equal,
            noise,
        },
        official: cfg.official,
    }
}

/// A deterministic cell index for a `(K, N_T)` pair.
fn cell_index(k: usize, n_traders: usize) -> u64 {
    // Distinct per pair; the seed derivation hashes it, so arithmetic overlap
    // between pairs is not a concern as long as the pair maps injectively.
    (k as u64) << 40 | n_traders as u64
}

/// Simulated MSE at one `(K, N_T)` cell.
fn mc_cell(cfg: &Config, k: usize, n_traders: usize) -> Result<(f64, f64, u64)> {
    let schedule = cfg.batch_schedule(cfg.participation_scale.realizations);
    let stats = pre_price_mse(
        Cell::new(
            cfg.master_seed,
            EXPERIMENT,
            cell_index(k, n_traders),
            &schedule,
        ),
        &market(cfg, k, n_traders, InterpretationNoise::Aggregated),
    )?;
    Ok((stats.mean(), stats.std_error(), stats.count()))
}

/// Run the participation curve, the marginal-gain table and the equivalence check.
pub fn run(cfg: &Config) -> Result<ParticipationScale> {
    let scale = &cfg.participation_scale;

    // Monte Carlo is run at the requested points and at their doubles, so the
    // simulated doubling gain has both arms.
    let mut mc_points: Vec<usize> = scale
        .mc_trader_values
        .iter()
        .flat_map(|&n| [n, 2 * n])
        .collect();
    mc_points.sort_unstable();
    mc_points.dedup();

    let mut curve = Vec::new();
    let mut marginal_gain = Vec::new();

    for &k in &scale.k_values {
        let floor = analytic_floor(k, scale.rho_s, scale.sigma_s);

        // Simulated cells for this source budget, keyed by trader count.
        let mut simulated: Vec<(usize, f64, f64, u64)> = Vec::new();
        for &n in &mc_points {
            let (mean, se, count) = mc_cell(cfg, k, n)?;
            simulated.push((n, mean, se, count));
        }
        let lookup = |n: usize| simulated.iter().find(|(m, ..)| *m == n).copied();

        for &n_traders in &scale.trader_values {
            let m = market(cfg, k, n_traders, InterpretationNoise::Aggregated);
            let analytic = analytic_pre_price_mse(&m);
            let hit = lookup(n_traders);
            curve.push(ScaleRow {
                k_sources: k,
                n_traders,
                rho_s: scale.rho_s,
                sigma_s: scale.sigma_s,
                interpretation_sigma: scale.interpretation_sigma,
                clientele_bias: scale.clientele_bias,
                analytic_floor: floor,
                analytic_mse: analytic,
                distance_to_floor: analytic_distance_to_floor(&m),
                relative_gap: analytic_relative_gap(&m),
                traders_per_source: n_traders as f64 / k as f64,
                full_coverage_assumed: n_traders < k,
                mc_mse: hit.map(|(_, mean, ..)| mean),
                mc_se: hit.map(|(_, _, se, _)| se),
                mc_realizations: hit.map(|(.., count)| count),
                z_gap: hit.map(|(_, mean, se, _)| (mean - analytic) / se),
            });

            let doubled = market(cfg, k, 2 * n_traders, InterpretationNoise::Aggregated);
            let analytic_doubled = analytic_pre_price_mse(&doubled);
            let gain = analytic_doubling_gain(&m);
            let mc = match (lookup(n_traders), lookup(2 * n_traders)) {
                (Some((_, a, se_a, _)), Some((_, b, se_b, _))) => {
                    let se = (se_a * se_a + se_b * se_b).sqrt();
                    Some((a - b, se))
                }
                _ => None,
            };
            marginal_gain.push(MarginalGainRow {
                k_sources: k,
                n_traders,
                analytic_mse: analytic,
                analytic_mse_doubled: analytic_doubled,
                analytic_gain: gain,
                analytic_gain_times_n: gain * n_traders as f64,
                mc_gain: mc.map(|(g, _)| g),
                mc_gain_se: mc.map(|(_, se)| se),
                mc_gain_resolved: mc.map(|(g, se)| g.abs() > 3.0 * se),
            });
        }
    }

    Ok(ParticipationScale {
        curve,
        marginal_gain,
        equivalence: equivalence(cfg)?,
    })
}

/// Check the aggregated closed-form belief representation against an explicit
/// per-trader one at tractable trader counts.
///
/// The large cells of this experiment rely on the aggregated representation
/// being exact rather than approximate. This is the check that earns that
/// reliance.
///
/// # The `N_T < K` corner
///
/// Agreement is expected only where the trader population can cover every
/// source. The canonical model uses equal *asymptotic* representation: each
/// source carries weight `1/K` whatever the trader count, so a market with fewer
/// traders than sources still prices off the full source aggregate. A literal
/// round-robin assignment cannot do that — with one trader and two sources, one
/// source goes unread and the price inherits that source's whole error rather
/// than the average of two.
///
/// Both rows are returned. `full_source_coverage` marks which is which, and only
/// the covered rows are gated.
pub fn equivalence(cfg: &Config) -> Result<Vec<EquivalenceRow>> {
    let scale = &cfg.participation_scale;
    let schedule = cfg.batch_schedule(scale.equivalence_realizations);
    let k = *scale.k_values.iter().min().unwrap_or(&2);
    let mut rows = Vec::new();
    for (idx, &n_traders) in scale.equivalence_trader_values.iter().enumerate() {
        let aggregated = pre_price_mse(
            Cell::new(
                cfg.master_seed,
                EQUIVALENCE_EXPERIMENT,
                idx as u64,
                &schedule,
            ),
            &market(cfg, k, n_traders, InterpretationNoise::Aggregated),
        )?;
        let explicit = pre_price_mse(
            Cell::new(
                cfg.master_seed,
                EQUIVALENCE_EXPERIMENT,
                1_000 + idx as u64,
                &schedule,
            ),
            &market(cfg, k, n_traders, InterpretationNoise::PerTrader),
        )?;
        let se = (aggregated.std_error().powi(2) + explicit.std_error().powi(2)).sqrt();
        rows.push(EquivalenceRow {
            n_traders,
            k_sources: k,
            full_source_coverage: n_traders >= k,
            aggregated_mse: aggregated.mean(),
            explicit_mse: explicit.mean(),
            analytic_mse: analytic_pre_price_mse(&market(
                cfg,
                k,
                n_traders,
                InterpretationNoise::Aggregated,
            )),
            z_gap: (aggregated.mean() - explicit.mean()) / se,
        });
    }
    Ok(rows)
}

/// The participation-scale experiment as an [`Experiment`].
#[derive(Debug, Clone, Copy, Default)]
pub struct ParticipationScaleExperiment;

impl Experiment for ParticipationScaleExperiment {
    type Output = ParticipationScale;

    fn name(&self) -> &'static str {
        "scale-participation"
    }

    fn run(&self, ctx: &SimulationContext<'_>) -> Result<Self::Output> {
        run(ctx.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> Config {
        let mut cfg = Config::quick();
        cfg.participation_scale.trader_values = vec![1, 10, 100, 1_000, 1_000_000];
        cfg.participation_scale.mc_trader_values = vec![1, 100, 1_000_000];
        cfg.participation_scale.k_values = vec![2, 10];
        cfg.participation_scale.realizations = 40_000;
        cfg.participation_scale.equivalence_trader_values = vec![1, 10];
        cfg.participation_scale.equivalence_realizations = 20_000;
        cfg
    }

    /// S2–S4: Monte Carlo tracks the finite-`N_T` curve at every scale, including
    /// one million traders.
    #[test]
    fn monte_carlo_tracks_the_finite_participation_curve() {
        let cfg = small();
        let result = run(&cfg).expect("runs");
        let validated: Vec<_> = result.curve.iter().filter(|r| r.z_gap.is_some()).collect();
        assert!(!validated.is_empty());
        for row in validated {
            let z = row.z_gap.unwrap_or(f64::NAN);
            assert!(
                z.abs() < 5.0,
                "K={} N_T={}: mc {:?} vs analytic {} ({z} SE)",
                row.k_sources,
                row.n_traders,
                row.mc_mse,
                row.analytic_mse
            );
        }
    }

    /// S5 and S7: the curve falls monotonically toward the floor and never below
    /// it, across six orders of magnitude.
    #[test]
    fn curve_declines_monotonically_to_the_floor() {
        let cfg = small();
        let result = run(&cfg).expect("runs");
        for &k in &cfg.participation_scale.k_values {
            let rows: Vec<_> = result.curve.iter().filter(|r| r.k_sources == k).collect();
            for pair in rows.windows(2) {
                assert!(
                    pair[1].analytic_mse < pair[0].analytic_mse,
                    "K={k}: MSE rose from N_T={} to {}",
                    pair[0].n_traders,
                    pair[1].n_traders
                );
            }
            let last = rows.last().expect("rows");
            assert!(last.analytic_mse > last.analytic_floor);
            assert!(
                last.relative_gap < 1e-5,
                "not saturated: {}",
                last.relative_gap
            );
        }
    }

    /// S6: the doubling gain decays toward zero, and `gain * N_T` is constant.
    #[test]
    fn marginal_gain_decays_as_one_over_participation() {
        let cfg = small();
        let result = run(&cfg).expect("runs");
        let expected = cfg.participation_scale.interpretation_sigma.powi(2) / 2.0;
        for row in &result.marginal_gain {
            assert!(
                (row.analytic_gain_times_n - expected).abs() < 1e-12,
                "N_T={}: gain*N was {}",
                row.n_traders,
                row.analytic_gain_times_n
            );
        }
        let biggest = result
            .marginal_gain
            .iter()
            .max_by_key(|r| r.n_traders)
            .expect("rows");
        assert!(biggest.analytic_gain < 1e-6);
    }

    /// S8: the source-count ordering holds at every participation level.
    #[test]
    fn source_ordering_holds_at_every_scale() {
        let cfg = small();
        let result = run(&cfg).expect("runs");
        for &n in &cfg.participation_scale.trader_values {
            let thin = result
                .curve
                .iter()
                .find(|r| r.k_sources == 2 && r.n_traders == n)
                .expect("row");
            let rich = result
                .curve
                .iter()
                .find(|r| r.k_sources == 10 && r.n_traders == n)
                .expect("row");
            assert!(
                thin.analytic_mse > rich.analytic_mse,
                "ordering broke at N_T={n}"
            );
        }
    }

    /// The aggregated representation is exact wherever the trader population can
    /// cover every source.
    #[test]
    fn aggregated_and_explicit_representations_agree_under_full_coverage() {
        let cfg = small();
        let rows = equivalence(&cfg).expect("runs");
        let covered: Vec<_> = rows.iter().filter(|r| r.full_source_coverage).collect();
        assert!(!covered.is_empty(), "no covered rows to check");
        for row in covered {
            assert!(
                row.z_gap.abs() < 5.0,
                "N_T={}: aggregated {} vs explicit {} ({} SE)",
                row.n_traders,
                row.aggregated_mse,
                row.explicit_mse,
                row.z_gap
            );
        }
    }

    /// Below one trader per source the two representations differ, and they
    /// differ in the direction the model says they should: a literal assignment
    /// leaves sources unread, so it prices worse.
    #[test]
    fn representations_diverge_when_traders_cannot_cover_the_sources() {
        let mut cfg = small();
        cfg.participation_scale.k_values = vec![4];
        cfg.participation_scale.equivalence_trader_values = vec![1, 2];
        let rows = equivalence(&cfg).expect("runs");
        assert!(rows.iter().all(|r| !r.full_source_coverage));
        for row in rows {
            assert!(
                row.explicit_mse > row.aggregated_mse,
                "N_T={}: a literal assignment should price worse, got {} vs {}",
                row.n_traders,
                row.explicit_mse,
                row.aggregated_mse
            );
        }
    }
}
