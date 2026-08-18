//! Gate: the analytic floor is the formula the model claims it is.
//!
//! `MSE_inf(K, rho_s) = sigma_s^2 [rho_s + (1 - rho_s)/K]` is checked against
//! its two closed-form special cases and its two comparative statics. These are
//! exact identities, so the tolerances are numerical, not statistical.

use crate::analytics::information_floor::analytic_floor;
use crate::error::Result;
use crate::validation::Check;

/// Check the closed-form identities and comparative statics of the floor.
pub fn check_analytic_floor() -> Result<Check> {
    let mut failures = Vec::new();

    // Independent sources: the floor is sigma_s^2 / K.
    for k in [1usize, 2, 5, 10, 50, 100] {
        for sigma in [0.45, 1.0, 1.8] {
            let got = analytic_floor(k, 0.0, sigma);
            let want = sigma * sigma / k as f64;
            if (got - want).abs() > 1e-12 {
                failures.push(format!("K={k}, sigma={sigma}: {got} != {want}"));
            }
        }
    }

    // As K grows with rho_s > 0, the floor tends to the common component.
    for rho in [0.1, 0.5, 0.9] {
        let sigma = 1.3;
        let limit = sigma * sigma * rho;
        let far = analytic_floor(1_000_000_000, rho, sigma);
        if (far - limit).abs() > 1e-8 {
            failures.push(format!("rho={rho}: K->inf gave {far}, expected {limit}"));
        }
    }

    // Correlation raises the floor; sources lower it.
    let (k, sigma) = (10usize, 1.0);
    let ordered = analytic_floor(k, 0.0, sigma) < analytic_floor(k, 0.5, sigma)
        && analytic_floor(k, 0.5, sigma) < analytic_floor(k, 0.9, sigma);
    if !ordered {
        failures.push("floor is not increasing in source correlation".to_string());
    }
    let falling = analytic_floor(2, 0.5, sigma) > analytic_floor(10, 0.5, sigma)
        && analytic_floor(10, 0.5, sigma) > analytic_floor(50, 0.5, sigma);
    if !falling {
        failures.push("floor is not decreasing in the source budget".to_string());
    }

    let detail = if failures.is_empty() {
        format!(
            "rho=0 gives sigma^2/K exactly; K->inf gives sigma^2 rho; MSE_inf(10,0.9)={:.4} > MSE_inf(10,0.5)={:.4} > MSE_inf(10,0)={:.4}",
            analytic_floor(10, 0.9, 1.0),
            analytic_floor(10, 0.5, 1.0),
            analytic_floor(10, 0.0, 1.0),
        )
    } else {
        failures.join("; ")
    };
    Ok(Check::new(
        "analytic source floor",
        failures.is_empty(),
        detail,
    ))
}
