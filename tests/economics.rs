//! Economic regression tests.
//!
//! These tests protect the *conclusions*, not the implementation. Each one
//! would fail if the model drifted back toward the superseded v2 behaviour in
//! which trader growth silently manufactured fundamental information, or if the
//! source layer stopped controlling the price-error floor.

use market_information_capacity::analytics::information_floor::analytic_floor;
use market_information_capacity::config::{
    Config, MarketConfig, OfficialSignalConfig, SourceConfig, TraderConfig,
};
use market_information_capacity::model::traders::{InterpretationNoise, Representation};
use market_information_capacity::simulation::monte_carlo::{Cell, pre_price_mse};
use market_information_capacity::simulation::{same_price, worlds};

const SEED: u64 = 20_260_915;

fn market(k: usize, n_traders: usize, rho_s: f64, sigma_s: f64, sigma_nu: f64) -> MarketConfig {
    MarketConfig {
        sources: SourceConfig {
            count: k,
            sigma: sigma_s,
            correlation: rho_s,
        },
        traders: TraderConfig {
            count: n_traders,
            interpretation_sigma: sigma_nu,
            clientele_bias: 0.0,
            representation: Representation::Equal,
            noise: InterpretationNoise::Aggregated,
        },
        official: OfficialSignalConfig::default(),
    }
}

fn schedule(realizations: usize) -> Vec<usize> {
    market_information_capacity::config::batch_schedule(realizations, 8)
}

/// Section 24: the analytic floor is exactly `sigma_s^2 [rho_s + (1-rho_s)/K]`,
/// including both closed-form special cases.
#[test]
fn analytic_floor_matches_its_closed_form() {
    for k in [1usize, 2, 5, 10, 50, 100, 1000] {
        for sigma_s in [0.45, 1.0, 1.8] {
            for rho_s in [0.0, 0.1, 0.5, 0.9, 1.0] {
                let expected = sigma_s * sigma_s * (rho_s + (1.0 - rho_s) / k as f64);
                assert!((analytic_floor(k, rho_s, sigma_s) - expected).abs() < 1e-12);
            }
            // Independent sources.
            assert!((analytic_floor(k, 0.0, sigma_s) - sigma_s * sigma_s / k as f64).abs() < 1e-12);
        }
    }
    // K -> infinity with positive correlation leaves the common component.
    for rho_s in [0.1, 0.5, 0.9] {
        let sigma_s = 1.3;
        let limit = sigma_s * sigma_s * rho_s;
        assert!((analytic_floor(1_000_000_000, rho_s, sigma_s) - limit).abs() < 1e-8);
    }
}

/// Section 25: trader duplication does not create fundamental information.
///
/// With a fixed budget of two correlated sources, growing the trader population
/// by two orders of magnitude must leave the price error pinned to the floor.
/// If someone reintroduces "one trader = one independent signal", the measured
/// MSE at `N_T = 10,000` collapses toward zero and this test fails.
#[test]
fn trader_duplication_does_not_create_fundamental_information() {
    let (k, rho_s, sigma_s, sigma_nu) = (2usize, 0.5, 1.0, 0.8);
    let floor = analytic_floor(k, rho_s, sigma_s);
    let sched = schedule(200_000);

    let mut measured = Vec::new();
    for (idx, n_traders) in [100usize, 1_000, 10_000].into_iter().enumerate() {
        let stats = pre_price_mse(
            Cell::new(SEED, "test-duplication", idx as u64, &sched),
            &market(k, n_traders, rho_s, sigma_s, sigma_nu),
        )
        .expect("run");
        measured.push((n_traders, stats.mean(), stats.std_error()));
    }

    for (n_traders, mse, se) in &measured {
        assert!(
            mse - floor > -4.0 * se,
            "N_T={n_traders}: MSE {mse} fell below the K={k} floor {floor}"
        );
        assert!(
            *mse > 0.9 * floor,
            "N_T={n_traders}: MSE {mse} collapsed relative to floor {floor}"
        );
    }

    // A hundredfold increase in participation buys at most a few percent.
    let (_, small, _) = measured[0];
    let (_, large, _) = measured[2];
    let relative_gain = (small - large) / small;
    assert!(
        relative_gain < 0.05,
        "growing N_T from 100 to 10,000 cut MSE by {relative_gain:.3}, which no finite source budget allows"
    );
}

/// Section 26: once participation has saturated, sources matter more than
/// traders. At high trader density, `MSE(N_T, 2K) < MSE(2 N_T, K)`.
#[test]
fn sources_matter_more_than_traders_after_participation_saturates() {
    let (k, n_traders, rho_s, sigma_s, sigma_nu) = (10usize, 5_000usize, 0.5, 1.0, 0.8);
    let sched = schedule(300_000);

    let double_traders = pre_price_mse(
        Cell::new(SEED, "test-saturation", 0, &sched),
        &market(k, 2 * n_traders, rho_s, sigma_s, sigma_nu),
    )
    .expect("run");
    let double_sources = pre_price_mse(
        Cell::new(SEED, "test-saturation", 1, &sched),
        &market(2 * k, n_traders, rho_s, sigma_s, sigma_nu),
    )
    .expect("run");

    let diff = double_traders.mean() - double_sources.mean();
    let se = (double_traders.std_error().powi(2) + double_sources.std_error().powi(2)).sqrt();
    assert!(
        diff > 4.0 * se,
        "doubling sources gave {} and doubling traders gave {} (difference {diff:.5}, SE {se:.5})",
        double_sources.mean(),
        double_traders.mean()
    );
}

/// Section 27: source correlation raises the floor, holding `K` and `sigma_s`
/// fixed — analytically and in simulation.
#[test]
fn source_correlation_raises_the_floor() {
    let (k, sigma_s) = (10usize, 1.0);
    let floors: Vec<f64> = [0.0, 0.5, 0.9]
        .iter()
        .map(|&rho| analytic_floor(k, rho, sigma_s))
        .collect();
    assert!(
        floors[0] < floors[1] && floors[1] < floors[2],
        "analytic floors were {floors:?}"
    );

    let sched = schedule(200_000);
    let mut simulated = Vec::new();
    for (idx, rho_s) in [0.0f64, 0.5, 0.9].into_iter().enumerate() {
        let stats = pre_price_mse(
            Cell::new(SEED, "test-correlation", idx as u64, &sched),
            &market(k, 5_000, rho_s, sigma_s, 0.8),
        )
        .expect("run");
        simulated.push((stats.mean(), stats.std_error()));
    }
    for pair in simulated.windows(2) {
        let (lo, se_lo) = pair[0];
        let (hi, se_hi) = pair[1];
        let se = (se_lo * se_lo + se_hi * se_hi).sqrt();
        assert!(
            hi - lo > 4.0 * se,
            "simulated MSE did not rise with correlation: {lo} then {hi} (SE {se})"
        );
    }
}

/// Section 28: the Fast Follower mechanism, not its canonical numbers.
#[test]
fn fast_follower_mechanism_survives() {
    let cfg = Config::quick();
    let rows = worlds::run(&cfg).expect("worlds run");
    let informative = rows
        .iter()
        .find(|r| r.world == "Informative")
        .expect("informative world");
    let fast_follower = rows
        .iter()
        .find(|r| r.world == "Fast Follower")
        .expect("fast follower world");

    // Many traders, thin and highly correlated pre-official information.
    assert!(fast_follower.n_traders >= 1_000);
    assert!(fast_follower.k_sources <= 2 && fast_follower.rho_s >= 0.75);

    // The pre-official price is weak in absolute terms and relative to a rich
    // information environment with the same trader count.
    assert!(
        fast_follower.pre_price_mse > 0.5,
        "pre MSE was {}",
        fast_follower.pre_price_mse
    );
    assert!(
        fast_follower.pre_price_mse > 10.0 * informative.pre_price_mse,
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
        "mean absolute revision was {}",
        fast_follower.mean_abs_revision
    );
    assert!(fast_follower.p90_abs_revision > fast_follower.p10_abs_revision);
}

/// Section 21 note: the official signal is a mechanical revision, so it is not
/// guaranteed to improve an already-accurate price. The Informative world must
/// be allowed to get worse.
#[test]
fn official_information_does_not_automatically_improve_price_quality() {
    let cfg = Config::quick();
    let rows = worlds::run(&cfg).expect("worlds run");
    let informative = rows
        .iter()
        .find(|r| r.world == "Informative")
        .expect("informative world");
    assert!(
        informative.post_improvement < 0.0,
        "the canonical Informative world should be harmed by the fixed-weight revision, got {}",
        informative.post_improvement
    );
}

/// Section 29: the same price carries different amounts of information.
#[test]
fn same_price_information_content_orders_by_information_capacity() {
    let mut cfg = Config::quick();
    cfg.same_price.realizations = 200_000;
    let rows = same_price::run(&cfg).expect("conditional run");

    let sd = |name: &str| {
        rows.iter()
            .find(|r| r.world == name)
            .unwrap_or_else(|| panic!("missing world {name}"))
            .sd_v
    };
    let informative = sd("Informative");
    let selected = sd("Selected Clientele");
    let fast_follower = sd("Fast Follower");

    assert!(
        informative < selected && selected < fast_follower,
        "conditional SDs were {informative:.3}, {selected:.3}, {fast_follower:.3}"
    );
    // The separations must be large relative to the sampling error of an SD,
    // which is about sd / sqrt(2n).
    for row in &rows {
        assert!(
            row.accepted_n > 5_000,
            "{} accepted only {}",
            row.world,
            row.accepted_n
        );
        let se = row.sd_v / (2.0 * row.accepted_n as f64).sqrt();
        assert!(se < 0.02, "{}: SD standard error {se}", row.world);
    }
}

/// A market with more traders than sources still cannot beat its floor, and a
/// market with more sources than traders still benefits from them under the
/// canonical equal-representation aggregation.
#[test]
fn floor_binds_on_both_sides_of_the_trader_source_ratio() {
    let sched = schedule(200_000);
    for (idx, (k, n_traders)) in [(2usize, 5_000usize), (100, 50)].into_iter().enumerate() {
        let floor = analytic_floor(k, 0.5, 1.0);
        let stats = pre_price_mse(
            Cell::new(SEED, "test-ratio", idx as u64, &sched),
            &market(k, n_traders, 0.5, 1.0, 0.8),
        )
        .expect("run");
        let expected = floor + 0.8 * 0.8 / n_traders as f64;
        let z = (stats.mean() - expected) / stats.std_error();
        assert!(
            z.abs() < 5.0,
            "K={k}, N_T={n_traders}: MSE {} vs expected {expected} ({z} SE)",
            stats.mean()
        );
    }
}
