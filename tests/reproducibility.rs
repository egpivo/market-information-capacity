//! Reproducibility contract.
//!
//! The same configuration and seed must produce the same canonical numbers,
//! bit for bit, no matter how the work was scheduled across threads.

use market_information_capacity::config::Config;
use market_information_capacity::simulation::{asymptote, same_price, worlds};

fn small(config: &mut Config) {
    config.asymptote.realizations = 20_000;
    config.asymptote.k_values = vec![10];
    config.asymptote.rho_values = vec![0.5];
    config.asymptote.trader_values = vec![1_000];
    config.worlds.realizations = 20_000;
    config.same_price.realizations = 40_000;
}

#[test]
fn identical_seed_and_config_reproduce_identical_results() {
    let mut cfg = Config::quick();
    small(&mut cfg);

    let first = worlds::run(&cfg).expect("first run");
    let second = worlds::run(&cfg).expect("second run");
    assert_eq!(
        first, second,
        "worlds results were not bitwise reproducible"
    );

    let a = asymptote::run(&cfg).expect("first run");
    let b = asymptote::run(&cfg).expect("second run");
    assert_eq!(a, b, "asymptote results were not bitwise reproducible");

    let p = same_price::run(&cfg).expect("first run");
    let q = same_price::run(&cfg).expect("second run");
    assert_eq!(p, q, "same-price results were not bitwise reproducible");
}

#[test]
fn a_different_master_seed_gives_different_draws() {
    let mut cfg = Config::quick();
    small(&mut cfg);
    let base = worlds::run(&cfg).expect("run");

    cfg.master_seed += 1;
    let shifted = worlds::run(&cfg).expect("run");

    assert_ne!(
        base[0].pre_price_mse.to_bits(),
        shifted[0].pre_price_mse.to_bits(),
        "changing the master seed must change the realized draws"
    );
    // But the same economics: both must sit on the same closed form.
    for (a, b) in base.iter().zip(shifted.iter()) {
        assert_eq!(a.analytic_pre_price_mse, b.analytic_pre_price_mse);
        let z = (a.pre_price_mse - b.pre_price_mse)
            / (a.pre_price_mse_se.powi(2) + b.pre_price_mse_se.powi(2)).sqrt();
        assert!(z.abs() < 5.0, "{}: seeds disagreed by {z} SE", a.world);
    }
}

#[test]
fn batch_count_does_not_change_the_economics() {
    let mut cfg = Config::quick();
    small(&mut cfg);
    cfg.batches = 1;
    let serial = worlds::run(&cfg).expect("serial run");

    cfg.batches = 13;
    let parallel = worlds::run(&cfg).expect("parallel run");

    for (a, b) in serial.iter().zip(parallel.iter()) {
        assert_eq!(a.realizations, b.realizations);
        let se = (a.pre_price_mse_se.powi(2) + b.pre_price_mse_se.powi(2)).sqrt();
        let z = (a.pre_price_mse - b.pre_price_mse) / se;
        assert!(
            z.abs() < 5.0,
            "{}: {} batches gave {} and {} batches gave {} ({z} SE apart)",
            a.world,
            1,
            a.pre_price_mse,
            13,
            b.pre_price_mse
        );
    }
}
