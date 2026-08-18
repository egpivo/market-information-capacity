//! When does adding traders stop helping?
//!
//! Participation growth acts only on the `sigma_nu^2 / N_T` term of the price
//! error. The relative distance of the finite-`N_T` MSE from its floor is
//!
//! ```text
//! gap(N_T) = (MSE(N_T) - MSE_inf) / MSE_inf = sigma_nu^2 / (N_T * MSE_inf)
//! ```
//!
//! in the clean benchmark (no clientele bias), so the trader count needed to
//! come within a relative gap `g` of the floor is
//!
//! ```text
//! N_T(g) = ceil( sigma_nu^2 / (g * MSE_inf) ).
//! ```
//!
//! These are descriptive convergence thresholds, not welfare optima.

use crate::analytics::information_floor::analytic_floor;

/// Analytic trader count needed to reach a relative gap `g` above the floor.
pub fn clean_threshold(k: usize, rho_s: f64, sigma_s: f64, sigma_nu: f64, gap: f64) -> Option<u64> {
    let floor = analytic_floor(k, rho_s, sigma_s);
    if gap <= 0.0 || !floor.is_finite() || floor <= 0.0 {
        return None;
    }
    let raw = sigma_nu * sigma_nu / (gap * floor);
    if !raw.is_finite() {
        return None;
    }
    Some(raw.ceil().max(1.0) as u64)
}

/// The first trader count on an ascending grid whose measured relative gap is
/// below `gap`.
///
/// `grid` must be ordered by trader count; the second element of each pair is
/// the measured relative gap at that trader count.
pub fn first_grid_threshold(grid: &[(usize, f64)], gap: f64) -> Option<usize> {
    grid.iter()
        .find(|(_, measured)| *measured < gap)
        .map(|(n, _)| *n)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The canonical v4 clean-benchmark thresholds, rho_s = 0.5, sigma_s = 1,
    /// interpretation noise 0.8: 18/86, 24/117, 26/126.
    #[test]
    fn reproduces_canonical_thresholds() {
        let cases = [(2usize, 18u64, 86u64), (10, 24, 117), (50, 26, 126)];
        for (k, five, one) in cases {
            assert_eq!(clean_threshold(k, 0.5, 1.0, 0.8, 0.05), Some(five), "K={k}");
            assert_eq!(clean_threshold(k, 0.5, 1.0, 0.8, 0.01), Some(one), "K={k}");
        }
    }

    #[test]
    fn tighter_gaps_need_more_traders() {
        let loose = clean_threshold(10, 0.5, 1.0, 0.8, 0.05).expect("finite");
        let tight = clean_threshold(10, 0.5, 1.0, 0.8, 0.01).expect("finite");
        assert!(tight > loose);
    }

    #[test]
    fn grid_threshold_picks_first_crossing() {
        let grid = [(50usize, 0.09), (100, 0.04), (250, 0.01)];
        assert_eq!(first_grid_threshold(&grid, 0.05), Some(100));
        assert_eq!(first_grid_threshold(&grid, 0.001), None);
    }
}
