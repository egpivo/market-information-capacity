//! Gate: the same price carries different amounts of information.
//!
//! Conditioning every world on the same narrow price band, the spread of the
//! latent value must widen monotonically as the information environment thins.

use crate::analytics::metrics::OnlineStats;
use crate::config::Config;
use crate::error::Result;
use crate::simulation::monte_carlo::{Cell, run_conditional};
use crate::validation::{Evidence, ValidationCheck, ValidationContext};

/// Stable experiment label used for seed derivation.
pub const EXPERIMENT: &str = "gate-same-price";

/// Check the conditional-spread ordering across worlds.
fn check_same_price(cfg: &Config) -> Result<Evidence> {
    let schedule = cfg.batch_schedule(cfg.validation.same_price_realizations);
    let band = cfg.same_price.band;
    let mut rows = Vec::new();
    for (idx, world) in cfg.worlds.presets.iter().enumerate() {
        let acc = run_conditional(
            Cell::new(cfg.master_seed, EXPERIMENT, idx as u64, &schedule),
            &world.market(cfg.official),
            band,
            cfg.same_price.retained_samples,
        )?;
        let accepted = acc.accepted.seen();
        let sorted = acc.accepted.into_sorted();
        let mut stats = OnlineStats::new();
        for &v in &sorted {
            stats.push(v);
        }
        rows.push((world.name.clone(), accepted, stats.std_dev()));
    }

    // Order the worlds by their information capacity: more sources and weaker
    // correlation means more capacity.
    let mut ordered: Vec<_> = rows.iter().cloned().enumerate().collect();
    ordered.sort_by(|(a, _), (b, _)| {
        let wa = &cfg.worlds.presets[*a];
        let wb = &cfg.worlds.presets[*b];
        (wa.rho_s / wa.sources as f64).total_cmp(&(wb.rho_s / wb.sources as f64))
    });

    // Standard error of a sample standard deviation is roughly sd / sqrt(2n).
    let widening = ordered.windows(2).all(|w| {
        let (_, (_, n_lo, sd_lo)) = &w[0];
        let (_, (_, n_hi, sd_hi)) = &w[1];
        let se =
            (sd_lo * sd_lo / (2.0 * *n_lo as f64) + sd_hi * sd_hi / (2.0 * *n_hi as f64)).sqrt();
        sd_hi - sd_lo > 3.0 * se
    });
    let enough = rows.iter().all(|(_, n, _)| *n >= 1_000);

    let summary: Vec<String> = ordered
        .iter()
        .map(|(_, (name, n, sd))| format!("{name}: sd {sd:.3} (n={n})"))
        .collect();
    Ok(Evidence::new(
        widening && enough,
        format!("band |P_OC| < {band}; {}", summary.join(", ")),
    ))
}

/// Gate: conditional spread widens as information capacity thins.
#[derive(Debug, Clone, Copy, Default)]
pub struct SamePriceConditional;

impl ValidationCheck for SamePriceConditional {
    fn name(&self) -> &'static str {
        "same-price conditional"
    }

    fn run(&self, ctx: &ValidationContext<'_>) -> Result<Evidence> {
        let _ = ctx;
        check_same_price(ctx.config)
    }
}
