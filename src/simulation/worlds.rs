//! The three-worlds experiment.
//!
//! The worlds are parameter regimes executed by the same simulator: nothing in
//! this module branches on which world it is running. Each world is carried
//! through the official-information boundary so that a market's pre-official
//! information content and its response to genuinely new information can be
//! read side by side.

use serde::Serialize;

use crate::analytics::information_floor::{
    analytic_floor, analytic_mean_abs_revision, analytic_post_price_mse, analytic_pre_price_mse,
};
use crate::analytics::metrics::quantile_sorted;
use crate::config::Config;
use crate::error::Result;
use crate::simulation::experiment::{Experiment, SimulationContext};
use crate::simulation::monte_carlo::{Cell, run_world};

/// Stable experiment label used for seed derivation.
pub const EXPERIMENT: &str = "worlds";

/// One world's results.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct WorldRow {
    /// World name.
    pub world: String,
    /// Trader count `N_T`.
    pub n_traders: usize,
    /// Source budget `K`.
    pub k_sources: usize,
    /// Source correlation.
    pub rho_s: f64,
    /// Source error scale.
    pub sigma_s: f64,
    /// Clientele tilt.
    pub bias: f64,
    /// Interpretation noise scale.
    pub interpretation_sigma: f64,
    /// Analytic asymptotic floor.
    pub analytic_floor: f64,
    /// Simulated pre-official price MSE.
    pub pre_price_mse: f64,
    /// Standard error of the pre-official MSE.
    pub pre_price_mse_se: f64,
    /// Exact pre-official price MSE.
    pub analytic_pre_price_mse: f64,
    /// MSE of an uninformed price of zero.
    pub uninformed_mse: f64,
    /// How much the pre-official price improved on an uninformed price.
    pub pre_improvement_vs_uninformed: f64,
    /// Simulated post-boundary price MSE.
    pub post_price_mse: f64,
    /// Standard error of the post-boundary MSE.
    pub post_price_mse_se: f64,
    /// Exact post-boundary price MSE.
    pub analytic_post_price_mse: f64,
    /// Pre minus post MSE. Negative where the mechanical revision hurts an
    /// already-accurate price.
    pub post_improvement: f64,
    /// Mean absolute price revision.
    pub mean_abs_revision: f64,
    /// Exact mean absolute price revision.
    pub analytic_mean_abs_revision: f64,
    /// 10th percentile of the absolute revision.
    pub p10_abs_revision: f64,
    /// 90th percentile of the absolute revision.
    pub p90_abs_revision: f64,
    /// Realizations behind the estimates.
    pub realizations: u64,
    /// Whether revision deciles came from a subsampled reservoir.
    pub revision_sample_truncated: bool,
}

/// Run every configured world.
pub fn run(cfg: &Config) -> Result<Vec<WorldRow>> {
    let schedule = cfg.batch_schedule(cfg.worlds.realizations);
    let mut rows = Vec::with_capacity(cfg.worlds.presets.len());
    for (idx, world) in cfg.worlds.presets.iter().enumerate() {
        let market = world.market(cfg.official);
        let acc = run_world(
            Cell::new(cfg.master_seed, EXPERIMENT, idx as u64, &schedule),
            &market,
            cfg.worlds.revision_samples,
        )?;
        let truncated = acc.revision_samples.truncated();
        let sorted = acc.revision_samples.into_sorted();
        rows.push(WorldRow {
            world: world.name.clone(),
            n_traders: world.traders,
            k_sources: world.sources,
            rho_s: world.rho_s,
            sigma_s: world.sigma_s,
            bias: world.clientele_bias,
            interpretation_sigma: world.interpretation_sigma,
            analytic_floor: analytic_floor(world.sources, world.rho_s, world.sigma_s),
            pre_price_mse: acc.pre_mse.mean(),
            pre_price_mse_se: acc.pre_mse.std_error(),
            analytic_pre_price_mse: analytic_pre_price_mse(&market),
            uninformed_mse: acc.uninformed_mse.mean(),
            pre_improvement_vs_uninformed: acc.uninformed_mse.mean() - acc.pre_mse.mean(),
            post_price_mse: acc.post_mse.mean(),
            post_price_mse_se: acc.post_mse.std_error(),
            analytic_post_price_mse: analytic_post_price_mse(&market),
            post_improvement: acc.pre_mse.mean() - acc.post_mse.mean(),
            mean_abs_revision: acc.abs_revision.mean(),
            analytic_mean_abs_revision: analytic_mean_abs_revision(&market),
            p10_abs_revision: quantile_sorted(&sorted, 0.10),
            p90_abs_revision: quantile_sorted(&sorted, 0.90),
            realizations: acc.pre_mse.count(),
            revision_sample_truncated: truncated,
        });
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_follower_shows_the_validated_signature() {
        let cfg = Config::quick();
        let rows = run(&cfg).expect("worlds run");
        let informative = &rows[0];
        let fast_follower = &rows[2];

        // Many traders, thin correlated information: weak pre-official price.
        assert_eq!(fast_follower.n_traders, 2500);
        assert!(fast_follower.k_sources <= 2);
        assert!(
            fast_follower.pre_price_mse > informative.pre_price_mse * 10.0,
            "fast follower pre MSE {} vs informative {}",
            fast_follower.pre_price_mse,
            informative.pre_price_mse
        );
        // Genuinely new official information moves it a long way.
        assert!(
            fast_follower.post_price_mse < 0.3 * fast_follower.pre_price_mse,
            "post {} vs pre {}",
            fast_follower.post_price_mse,
            fast_follower.pre_price_mse
        );
        assert!(
            fast_follower.mean_abs_revision > 0.3,
            "revision {}",
            fast_follower.mean_abs_revision
        );
    }

    #[test]
    fn simulated_worlds_match_their_closed_forms() {
        let cfg = Config::quick();
        for row in run(&cfg).expect("worlds run") {
            let z = (row.pre_price_mse - row.analytic_pre_price_mse) / row.pre_price_mse_se;
            assert!(z.abs() < 5.0, "{}: pre MSE off by {z} se", row.world);
            let z_post = (row.post_price_mse - row.analytic_post_price_mse) / row.post_price_mse_se;
            assert!(
                z_post.abs() < 5.0,
                "{}: post MSE off by {z_post} se",
                row.world
            );
        }
    }
}

/// The three-worlds experiment as an [`Experiment`].
#[derive(Debug, Clone, Copy, Default)]
pub struct WorldsExperiment;

impl Experiment for WorldsExperiment {
    type Output = Vec<WorldRow>;

    fn name(&self) -> &'static str {
        "worlds"
    }

    fn run(&self, ctx: &SimulationContext<'_>) -> Result<Self::Output> {
        run(ctx.config)
    }
}
