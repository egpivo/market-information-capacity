//! Canonical experiments.
//!
//! Every experiment is a thin layer over [`monte_carlo`]: it enumerates
//! parameter cells, runs deterministic batched Monte Carlo, and pairs each
//! simulated number with its analytic counterpart. No experiment branches on
//! which world or regime it is running.

pub mod asymptote;
pub mod monte_carlo;
pub mod same_price;
pub mod sensitivity;
pub mod traders_vs_sources;
pub mod worlds;

pub use asymptote::AsymptoteRow;
pub use same_price::SamePriceRow;
pub use sensitivity::{PhaseRow, Sensitivity, SensitivityRow};
pub use traders_vs_sources::{ConvergenceRow, DoublingRow, GridRow, TradersVsSources};
pub use worlds::WorldRow;
