//! Gates for the participation-scale experiment.
//!
//! These run inside `scale-participation` rather than inside the canonical
//! `validate` gate, so the existing gate's output is unchanged. They check the
//! properties that make the scale result trustworthy: that the closed-form
//! finite-`N_T` curve is right, that Monte Carlo tracks it at every scale
//! including one million traders, that the curve saturates onto the existing
//! information floor rather than through it, and that the marginal return to
//! participation decays to zero without the source ordering breaking.

use crate::config::Config;
use crate::simulation::participation_scale::ParticipationScale;
use crate::validation::Check;

/// Run every scale gate against a completed experiment.
pub fn run_scale_gates(cfg: &Config, result: &ParticipationScale) -> Vec<Check> {
    vec![
        s1_finite_curve(cfg, result),
        s2_to_s4_monte_carlo_parity(cfg, result),
        s5_approaches_floor(result),
        s6_marginal_gain_vanishes(cfg, result),
        s7_no_deterioration(result),
        s8_source_ordering(cfg, result),
        s_equivalence(cfg, result),
    ]
}

/// S1: the finite-`N_T` curve equals floor + tilt² + `sigma_nu²/N_T` exactly.
fn s1_finite_curve(cfg: &Config, result: &ParticipationScale) -> Check {
    let scale = &cfg.participation_scale;
    let bias2 = scale.clientele_bias * scale.clientele_bias;
    let sigma_nu2 = scale.interpretation_sigma * scale.interpretation_sigma;
    let mut worst = 0.0f64;
    for row in &result.curve {
        let expected = row.analytic_floor + bias2 + sigma_nu2 / row.n_traders as f64;
        worst = worst.max((row.analytic_mse - expected).abs());
    }
    Check::new(
        "S1 finite-N analytic curve",
        worst < 1e-12,
        format!(
            "MSE(N_T) = floor + b^2 + sigma_nu^2/N_T reproduced to {worst:.2e} across {} points",
            result.curve.len()
        ),
    )
}

/// S2-S4: Monte Carlo tracks the curve at small, medium and one-million scale.
fn s2_to_s4_monte_carlo_parity(cfg: &Config, result: &ParticipationScale) -> Check {
    let tolerance = cfg.participation_scale.parity_z_tolerance;
    let validated: Vec<_> = result
        .curve
        .iter()
        .filter_map(|r| r.z_gap.map(|z| (r.k_sources, r.n_traders, z)))
        .collect();
    let worst = validated
        .iter()
        .fold((0usize, 0usize, 0.0f64), |acc, &(k, n, z)| {
            if z.abs() > acc.2 {
                (k, n, z.abs())
            } else {
                acc
            }
        });
    let failures = validated
        .iter()
        .filter(|(_, _, z)| z.abs() > tolerance)
        .count();
    let largest_n = validated.iter().map(|(_, n, _)| *n).max().unwrap_or(0);
    Check::new(
        "S2-S4 monte carlo tracks the curve",
        failures == 0 && !validated.is_empty(),
        format!(
            "{} validated cells up to N_T={largest_n}; largest deviation {:.2} SE at K={}, N_T={} (tolerance {tolerance:.1} SE)",
            validated.len(),
            worst.2,
            worst.0,
            worst.1
        ),
    )
}

/// S5: the curve approaches the existing analytic floor from above.
fn s5_approaches_floor(result: &ParticipationScale) -> Check {
    let mut worst_gap = 0.0f64;
    let mut below = 0usize;
    let mut n_at_worst = 0usize;
    for row in &result.curve {
        if row.analytic_mse < row.analytic_floor {
            below += 1;
        }
    }
    let mut k_values: Vec<usize> = result.curve.iter().map(|r| r.k_sources).collect();
    k_values.sort_unstable();
    k_values.dedup();
    for k in &k_values {
        if let Some(last) = result
            .curve
            .iter()
            .filter(|r| r.k_sources == *k)
            .max_by_key(|r| r.n_traders)
        {
            if last.relative_gap > worst_gap {
                worst_gap = last.relative_gap;
                n_at_worst = last.n_traders;
            }
        }
    }
    Check::new(
        "S5 approaches the information floor",
        below == 0 && worst_gap < 1e-4,
        format!(
            "no point falls below its floor; largest relative gap at N_T={n_at_worst} is {worst_gap:.2e}"
        ),
    )
}

/// S6: the doubling gain decays to zero and `gain * N_T` is constant.
fn s6_marginal_gain_vanishes(cfg: &Config, result: &ParticipationScale) -> Check {
    let expected = cfg.participation_scale.interpretation_sigma.powi(2) / 2.0;
    let worst_product = result.marginal_gain.iter().fold(0.0f64, |a, r| {
        a.max((r.analytic_gain_times_n - expected).abs())
    });
    let largest = result
        .marginal_gain
        .iter()
        .max_by_key(|r| r.n_traders)
        .map(|r| (r.n_traders, r.analytic_gain))
        .unwrap_or((0, f64::NAN));
    Check::new(
        "S6 marginal gain vanishes",
        worst_product < 1e-12 && largest.1 < 1e-6,
        format!(
            "gain*N_T constant at {expected} to {worst_product:.2e}; gain at N_T={} is {:.3e}",
            largest.0, largest.1
        ),
    )
}

/// S7: no systematic deterioration as participation grows.
fn s7_no_deterioration(result: &ParticipationScale) -> Check {
    let mut k_values: Vec<usize> = result.curve.iter().map(|r| r.k_sources).collect();
    k_values.sort_unstable();
    k_values.dedup();
    let mut violations = Vec::new();
    for k in &k_values {
        let rows: Vec<_> = result.curve.iter().filter(|r| r.k_sources == *k).collect();
        for pair in rows.windows(2) {
            if pair[1].analytic_mse >= pair[0].analytic_mse {
                violations.push(format!("K={k} at N_T={}", pair[1].n_traders));
            }
        }
    }
    Check::new(
        "S7 no deterioration with scale",
        violations.is_empty(),
        if violations.is_empty() {
            format!(
                "MSE strictly decreasing in N_T for every source budget across {} points",
                result.curve.len()
            )
        } else {
            violations.join(", ")
        },
    )
}

/// S8: the source-count ordering holds at every participation level.
fn s8_source_ordering(cfg: &Config, result: &ParticipationScale) -> Check {
    let mut k_values = cfg.participation_scale.k_values.clone();
    k_values.sort_unstable();
    let mut violations = 0usize;
    for &n in &cfg.participation_scale.trader_values {
        let mut previous = f64::INFINITY;
        for &k in &k_values {
            if let Some(row) = result
                .curve
                .iter()
                .find(|r| r.k_sources == k && r.n_traders == n)
            {
                if row.analytic_mse >= previous {
                    violations += 1;
                }
                previous = row.analytic_mse;
            }
        }
    }
    Check::new(
        "S8 source ordering holds at every scale",
        violations == 0,
        format!(
            "MSE strictly decreasing in K at all {} participation levels",
            cfg.participation_scale.trader_values.len()
        ),
    )
}

/// The aggregated belief representation used for the large cells is exact.
fn s_equivalence(cfg: &Config, result: &ParticipationScale) -> Check {
    let tolerance = cfg.participation_scale.parity_z_tolerance;
    let covered: Vec<_> = result
        .equivalence
        .iter()
        .filter(|r| r.full_source_coverage)
        .collect();
    let worst = covered.iter().fold((0usize, 0.0f64), |acc, r| {
        if r.z_gap.abs() > acc.1 {
            (r.n_traders, r.z_gap.abs())
        } else {
            acc
        }
    });
    let ok = !covered.is_empty() && covered.iter().all(|r| r.z_gap.abs() < tolerance);
    let uncovered = result.equivalence.len() - covered.len();
    Check::new(
        "aggregated representation is exact",
        ok,
        format!(
            "{} trader counts with full source coverage agree with an explicit per-trader implementation; largest deviation {:.2} SE at N_T={}. {uncovered} row(s) below one trader per source are reported rather than gated: there the two representations differ by construction",
            covered.len(),
            worst.1,
            worst.0
        ),
    )
}
