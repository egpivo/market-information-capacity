//! Analytic results and streaming statistics.
//!
//! Everything in this module is closed form or online: nothing here consumes a
//! random stream. The analytic results are used three ways — as documented
//! theory, as validation targets for the Monte Carlo engine, and as reference
//! columns written next to every simulated number in the canonical CSVs.

pub mod convergence;
pub mod information_floor;
pub mod metrics;

pub use convergence::{clean_threshold, first_grid_threshold};
pub use information_floor::{
    analytic_distance_to_floor, analytic_doubling_gain, analytic_floor, analytic_mean_abs_revision,
    analytic_post_price_mse, analytic_pre_price_mse, analytic_relative_gap, approx_conditional_sd,
    floor_of,
};
pub use metrics::{McSummary, OnlineStats, SampleReservoir, quantile_sorted};
