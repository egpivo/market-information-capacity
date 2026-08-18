//! The analytic finite-source information floor.
//!
//! This is the central analytic object of the repository, implemented as a
//! first-class function rather than left in the documentation.
//!
//! Under equal source representation the price converges, as the trader
//! population grows, to the equal-weighted source aggregate. Writing
//! `s_k = V + sigma_s (sqrt(rho_s) u + sqrt(1 - rho_s) eta_k)`, the mean of `K`
//! sources has error `sigma_s (sqrt(rho_s) u + sqrt(1 - rho_s) eta_bar)` with
//! `Var(eta_bar) = 1/K`, hence
//!
//! ```text
//! MSE_inf(K, rho_s) = sigma_s^2 * [ rho_s + (1 - rho_s) / K ].
//! ```
//!
//! The trader count `N_T` does not appear.

use crate::config::MarketConfig;

/// The asymptotic price-error floor `MSE_inf(K, rho_s)` at source scale `sigma_s`.
///
/// # Panics
///
/// Never panics; `k == 0` yields infinity, which no validated configuration can
/// produce.
#[inline]
pub fn analytic_floor(k: usize, rho_s: f64, sigma_s: f64) -> f64 {
    if k == 0 {
        return f64::INFINITY;
    }
    sigma_s * sigma_s * (rho_s + (1.0 - rho_s) / k as f64)
}

/// The floor implied by a market configuration.
#[inline]
pub fn floor_of(cfg: &MarketConfig) -> f64 {
    analytic_floor(
        cfg.sources.count,
        cfg.sources.correlation,
        cfg.sources.sigma,
    )
}

/// Exact finite-`N_T` pre-official price MSE for the canonical market.
///
/// `P_OC - V = sigma_s (sqrt(rho) u + sqrt(1-rho) eta_bar) + b + nu_bar`, whose
/// three terms are independent, so
///
/// ```text
/// MSE(N_T, K, rho_s) = MSE_inf(K, rho_s) + b^2 + sigma_nu^2 / N_T.
/// ```
///
/// The `b^2` and `sigma_nu^2 / N_T` terms are what participation growth can act
/// on; the first term is what it cannot.
pub fn analytic_pre_price_mse(cfg: &MarketConfig) -> f64 {
    let bias = cfg.traders.clientele_bias;
    let sigma_nu = cfg.traders.interpretation_sigma;
    floor_of(cfg) + bias * bias + sigma_nu * sigma_nu / cfg.traders.count as f64
}

/// Exact post-boundary price MSE under the fixed-weight official revision.
///
/// `P_post - V = (1 - w)(P_pre - V) + w * sigma_o * epsilon`, so
///
/// ```text
/// MSE_post = (1 - w)^2 * MSE_pre + w^2 * sigma_o^2.
/// ```
pub fn analytic_post_price_mse(cfg: &MarketConfig) -> f64 {
    let w = cfg.official.weight;
    let sigma_o = cfg.official.sigma;
    (1.0 - w) * (1.0 - w) * analytic_pre_price_mse(cfg) + w * w * sigma_o * sigma_o
}

/// Exact mean absolute price revision across the official boundary.
///
/// The revision is `w * (official - P_pre)`, a normal with mean `-w * b` and
/// variance `w^2 (sigma_o^2 + Var(P_pre - V))`. For `X ~ N(mu, sigma^2)`,
/// `E|X| = sigma sqrt(2/pi) exp(-mu^2 / 2 sigma^2) + mu (1 - 2 Phi(-mu/sigma))`.
pub fn analytic_mean_abs_revision(cfg: &MarketConfig) -> f64 {
    let w = cfg.official.weight;
    let bias = cfg.traders.clientele_bias;
    let sigma_nu = cfg.traders.interpretation_sigma;
    let pre_variance = floor_of(cfg) + sigma_nu * sigma_nu / cfg.traders.count as f64;
    let mu = -w * bias;
    let variance = w * w * (cfg.official.sigma * cfg.official.sigma + pre_variance);
    folded_normal_mean(mu, variance.sqrt())
}

/// Mean of `|X|` for `X ~ N(mu, sigma^2)`.
pub fn folded_normal_mean(mu: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        return mu.abs();
    }
    let z = mu / sigma;
    sigma * (2.0 / std::f64::consts::PI).sqrt() * (-0.5 * z * z).exp()
        + mu * (1.0 - 2.0 * standard_normal_cdf(-z))
}

/// Standard normal CDF via the error function identity.
pub fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function, accurate to near double precision on the range this crate
/// uses.
///
/// Uses the everywhere-positive series
/// `erf(x) = (2/sqrt(pi)) e^{-x^2} sum_n 2^n x^{2n+1} / (1*3*...*(2n+1))`,
/// which has no cancellation, and saturates beyond `|x| = 6` where `erf`
/// differs from `±1` by less than 1e-17.
fn erf(x: f64) -> f64 {
    if !x.is_finite() {
        return if x.is_nan() { f64::NAN } else { x.signum() };
    }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    if x >= 6.0 {
        return sign;
    }
    let x2 = x * x;
    let mut term = x;
    let mut sum = x;
    let mut n = 0.0f64;
    loop {
        n += 1.0;
        term *= 2.0 * x2 / (2.0 * n + 1.0);
        sum += term;
        if term <= sum * 1e-18 {
            break;
        }
    }
    sign * 2.0 / std::f64::consts::PI.sqrt() * (-x2).exp() * sum
}

/// Exact standard deviation of `V` conditional on `|P_OC| < band`.
///
/// `(V, P_OC)` is jointly normal with `Var(V) = 1`, `Cov(V, P_OC) = 1` and
/// `Var(P_OC) = 1 + MSE_inf + sigma_nu^2/N_T`. Writing `beta = 1 / Var(P_OC)`,
/// the regression residual has variance `1 - beta`, and conditioning on a
/// narrow band around zero contributes `beta^2 * Var(P_OC | band)`. For a band
/// narrow relative to `sd(P_OC)` the truncated price is close to uniform, so
/// `Var(P_OC | band) ~ (2 * band)^2 / 12`.
///
/// This closed form is used as an independent cross-check on the conditional
/// Monte Carlo, not as a substitute for it.
pub fn approx_conditional_sd(cfg: &MarketConfig, band: f64) -> f64 {
    let sigma_nu = cfg.traders.interpretation_sigma;
    let price_variance = 1.0 + floor_of(cfg) + sigma_nu * sigma_nu / cfg.traders.count as f64;
    let beta = 1.0 / price_variance;
    let residual = 1.0 - beta;
    let band_variance = (2.0 * band) * (2.0 * band) / 12.0;
    (residual + beta * beta * band_variance).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn independent_sources_give_sigma_squared_over_k() {
        for k in [1usize, 2, 5, 10, 50, 100] {
            for sigma in [0.45, 1.0, 1.7] {
                assert_relative_eq!(
                    analytic_floor(k, 0.0, sigma),
                    sigma * sigma / k as f64,
                    epsilon = 1e-12
                );
            }
        }
    }

    #[test]
    fn floor_tends_to_common_component_as_k_grows() {
        let (rho, sigma) = (0.35, 1.2);
        let limit = sigma * sigma * rho;
        let mut previous = f64::INFINITY;
        for k in [1usize, 10, 100, 10_000, 1_000_000] {
            let floor = analytic_floor(k, rho, sigma);
            assert!(floor > limit, "floor must stay above the common component");
            assert!(floor < previous, "floor must fall in K");
            previous = floor;
        }
        assert_relative_eq!(
            analytic_floor(1_000_000_000, rho, sigma),
            limit,
            epsilon = 1e-8
        );
    }

    #[test]
    fn correlation_raises_the_floor() {
        let (k, sigma) = (10usize, 1.0);
        let low = analytic_floor(k, 0.0, sigma);
        let mid = analytic_floor(k, 0.5, sigma);
        let high = analytic_floor(k, 0.9, sigma);
        assert!(low < mid && mid < high, "{low} {mid} {high}");
    }

    #[test]
    fn folded_normal_mean_matches_known_case() {
        // E|X| for X ~ N(0, 1) is sqrt(2/pi).
        assert_relative_eq!(
            folded_normal_mean(0.0, 1.0),
            (2.0 / std::f64::consts::PI).sqrt(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn normal_cdf_is_calibrated() {
        assert_relative_eq!(standard_normal_cdf(0.0), 0.5, epsilon = 1e-9);
        assert_relative_eq!(
            standard_normal_cdf(1.96),
            0.975_002_104_851_780,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            standard_normal_cdf(-1.96),
            0.024_997_895_148_220,
            epsilon = 1e-12
        );
        assert_relative_eq!(standard_normal_cdf(-8.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(standard_normal_cdf(8.0), 1.0, epsilon = 1e-12);
        // Symmetry to machine precision.
        for z in [0.1, 0.5, 1.0, 2.5, 4.0] {
            assert_relative_eq!(
                standard_normal_cdf(z) + standard_normal_cdf(-z),
                1.0,
                epsilon = 1e-14
            );
        }
    }
}
