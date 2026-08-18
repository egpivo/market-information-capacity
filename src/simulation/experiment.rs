//! The experiment lifecycle.
//!
//! Every canonical experiment shares one shape: read the configuration, run
//! deterministic Monte Carlo, return a structured result that can be serialised.
//! [`Experiment`] captures exactly that and nothing more.
//!
//! The output is an **associated type**, not a common row or dataframe type.
//! Experiments legitimately produce different shapes — the asymptote benchmark
//! produces one table, traders-versus-sources produces three — and flattening
//! them into a single container to satisfy a trait would lose structure for no
//! gain. Where the files land stays an output concern and lives in
//! [`crate::pipeline`].

use serde::Serialize;

use crate::config::Config;
use crate::error::Result;

/// Everything an experiment needs in order to run.
#[derive(Debug, Clone, Copy)]
pub struct SimulationContext<'a> {
    /// The configuration, including the master seed.
    pub config: &'a Config,
}

impl<'a> SimulationContext<'a> {
    /// Build a context around a configuration.
    pub fn new(config: &'a Config) -> Self {
        Self { config }
    }
}

/// One canonical experiment.
pub trait Experiment {
    /// The structured result this experiment produces.
    type Output: Serialize;

    /// Stable name, matching the CLI subcommand.
    fn name(&self) -> &'static str;

    /// Run the experiment.
    fn run(&self, ctx: &SimulationContext<'_>) -> Result<Self::Output>;
}
