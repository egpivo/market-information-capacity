//! Streaming statistics.
//!
//! Publication-scale runs touch hundreds of millions of realizations, so no
//! experiment materialises a full sample matrix. Means, variances, mean squared
//! errors and standard errors are accumulated with Welford's algorithm, which
//! is numerically stable and mergeable across parallel batches.

use serde::Serialize;

/// Numerically stable streaming mean/variance accumulator.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct OnlineStats {
    count: u64,
    mean: f64,
    m2: f64,
}

impl OnlineStats {
    /// An empty accumulator.
    pub const fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Add one observation.
    #[inline]
    pub fn push(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    /// Number of observations.
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Sample mean, or zero when empty.
    #[inline]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Unbiased sample variance (`ddof = 1`), or `NaN` with fewer than two
    /// observations.
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            return f64::NAN;
        }
        self.m2 / (self.count - 1) as f64
    }

    /// Sample standard deviation (`ddof = 1`).
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Standard error of the mean.
    pub fn std_error(&self) -> f64 {
        if self.count < 2 {
            return f64::NAN;
        }
        (self.variance() / self.count as f64).sqrt()
    }

    /// Merge another accumulator (Chan-Golub-LeVeque parallel update).
    ///
    /// Merging is associative up to floating point, and callers always merge in
    /// a fixed batch order, so results do not depend on thread scheduling.
    pub fn merge(&mut self, other: &Self) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = *other;
            return;
        }
        let n_a = self.count as f64;
        let n_b = other.count as f64;
        let n = n_a + n_b;
        let delta = other.mean - self.mean;
        self.mean += delta * n_b / n;
        self.m2 += other.m2 + delta * delta * n_a * n_b / n;
        self.count += other.count;
    }
}

/// Summary of a Monte Carlo mean, as written to CSV/JSON.
#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct McSummary {
    /// Number of realizations.
    pub n: u64,
    /// Monte Carlo estimate of the mean.
    pub mean: f64,
    /// Standard error of that estimate.
    pub std_error: f64,
}

impl From<OnlineStats> for McSummary {
    fn from(stats: OnlineStats) -> Self {
        Self {
            n: stats.count(),
            mean: stats.mean(),
            std_error: stats.std_error(),
        }
    }
}

/// A deterministically bounded sample of retained values.
///
/// Conditional experiments (quantiles of `V` given a price band, revision
/// deciles) need order statistics rather than moments. Values are retained
/// exactly up to `capacity`; beyond it the reservoir switches to Algorithm R
/// driven by the batch's own deterministic stream, and records that truncation
/// happened so no result silently reports a censored sample.
#[derive(Debug, Clone)]
pub struct SampleReservoir {
    values: Vec<f64>,
    capacity: usize,
    seen: u64,
    truncated: bool,
}

impl SampleReservoir {
    /// A reservoir retaining up to `capacity` values.
    pub fn new(capacity: usize) -> Self {
        Self {
            values: Vec::new(),
            capacity,
            seen: 0,
            truncated: false,
        }
    }

    /// Offer a value to the reservoir.
    pub fn push<R: rand::Rng + ?Sized>(&mut self, x: f64, rng: &mut R) {
        self.seen += 1;
        if self.values.len() < self.capacity {
            self.values.push(x);
            return;
        }
        self.truncated = true;
        let idx = rng.random_range(0..self.seen);
        if (idx as usize) < self.capacity {
            self.values[idx as usize] = x;
        }
    }

    /// How many values were offered.
    #[inline]
    pub fn seen(&self) -> u64 {
        self.seen
    }

    /// Whether the reservoir had to drop values.
    #[inline]
    pub fn truncated(&self) -> bool {
        self.truncated
    }

    /// The retained values.
    #[inline]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Concatenate another reservoir, preserving offer counts.
    ///
    /// Callers merge batches in fixed index order, so the merged sample is
    /// reproducible.
    pub fn merge(&mut self, other: &Self) {
        self.values.extend_from_slice(&other.values);
        self.seen += other.seen;
        self.truncated |= other.truncated;
    }

    /// Consume the reservoir, returning its values sorted ascending.
    pub fn into_sorted(self) -> Vec<f64> {
        let mut values = self.values;
        values.sort_by(f64::total_cmp);
        values
    }
}

/// Linear-interpolated quantile of a slice that is already sorted ascending.
///
/// Matches the default (`linear`) definition used by NumPy, so Rust and the
/// Python prototype report the same conditional intervals.
pub fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let q = q.clamp(0.0, 1.0);
    let h = (sorted.len() - 1) as f64 * q;
    let lo = h.floor();
    let hi = h.ceil();
    if lo == hi {
        return sorted[h as usize];
    }
    let frac = h - lo;
    sorted[lo as usize] * (1.0 - frac) + sorted[hi as usize] * frac
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn welford_matches_two_pass() {
        let data: Vec<f64> = (0..1000)
            .map(|i| (i as f64 * 0.37).sin() * 3.0 + 1.5)
            .collect();
        let mut stats = OnlineStats::new();
        for &x in &data {
            stats.push(x);
        }
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        assert_relative_eq!(stats.mean(), mean, epsilon = 1e-12);
        assert_relative_eq!(stats.variance(), var, epsilon = 1e-10);
    }

    #[test]
    fn merge_matches_single_pass() {
        let data: Vec<f64> = (0..977).map(|i| (i as f64 * 0.11).cos() * 2.0).collect();
        let mut whole = OnlineStats::new();
        for &x in &data {
            whole.push(x);
        }
        let mut merged = OnlineStats::new();
        for chunk in data.chunks(97) {
            let mut part = OnlineStats::new();
            for &x in chunk {
                part.push(x);
            }
            merged.merge(&part);
        }
        assert_eq!(merged.count(), whole.count());
        assert_relative_eq!(merged.mean(), whole.mean(), epsilon = 1e-12);
        assert_relative_eq!(merged.variance(), whole.variance(), epsilon = 1e-10);
    }

    #[test]
    fn quantiles_match_numpy_linear_definition() {
        let sorted = [1.0, 2.0, 3.0, 4.0];
        assert_relative_eq!(quantile_sorted(&sorted, 0.0), 1.0);
        assert_relative_eq!(quantile_sorted(&sorted, 1.0), 4.0);
        assert_relative_eq!(quantile_sorted(&sorted, 0.5), 2.5);
        // numpy.quantile([1,2,3,4], 0.05) == 1.15
        assert_relative_eq!(quantile_sorted(&sorted, 0.05), 1.15, epsilon = 1e-12);
    }
}
