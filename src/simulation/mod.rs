//! Canonical experiments.
//!
//! Every experiment is a thin layer over [`monte_carlo`]: it enumerates
//! parameter cells, runs deterministic batched Monte Carlo, and pairs each
//! simulated number with its analytic counterpart. No experiment branches on
//! which world or regime it is running.
//!
//! Each one implements [`Experiment`], whose associated `Output` type preserves
//! the natural shape of its results rather than forcing every experiment through
//! a common table.

pub mod asymptote;
pub mod experiment;
pub mod monte_carlo;
pub mod same_price;
pub mod sensitivity;
pub mod traders_vs_sources;
pub mod worlds;

pub use asymptote::{AsymptoteExperiment, AsymptoteRow};
pub use experiment::{Experiment, SimulationContext};
pub use same_price::{SamePriceExperiment, SamePriceRow};
pub use sensitivity::{PhaseRow, Sensitivity, SensitivityExperiment, SensitivityRow};
pub use traders_vs_sources::{
    ConvergenceRow, DoublingRow, GridRow, TradersVsSources, TradersVsSourcesExperiment,
};
pub use worlds::{WorldRow, WorldsExperiment};
