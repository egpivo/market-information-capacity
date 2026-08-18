//! The research-integrity gate.
//!
//! Each submodule owns one class of check and returns [`Check`] values that
//! record what was measured, not merely whether something passed. The gate
//! fails the process (non-zero exit) if any required check fails, so a silent
//! model regression cannot reach a published result.

pub mod analytic_floor;
pub mod determinism;
pub mod fast_follower;
pub mod finite_source;
pub mod gate;
pub mod hierarchy;
pub mod same_price;

use serde::Serialize;

use crate::config::Config;
use crate::error::Result;

/// Everything a validation check needs in order to run.
#[derive(Debug, Clone, Copy)]
pub struct ValidationContext<'a> {
    /// The configuration, including the master seed and the gate's scale.
    pub config: &'a Config,
}

impl<'a> ValidationContext<'a> {
    /// Build a context around a configuration.
    pub fn new(config: &'a Config) -> Self {
        Self { config }
    }
}

/// What a check measured, before it is labelled with a name.
#[derive(Debug, Clone)]
pub struct Evidence {
    /// Whether the property held.
    pub passed: bool,
    /// Human-readable evidence, including the measured numbers.
    pub detail: String,
}

impl Evidence {
    /// Build evidence from a condition and its detail.
    pub fn new(passed: bool, detail: impl Into<String>) -> Self {
        Self {
            passed,
            detail: detail.into(),
        }
    }
}

/// One property of the model that must hold before a result is published.
///
/// Checks are heterogeneous, are nowhere near a performance hot path, and are
/// enumerated once per run, so this is the one place in the crate where dynamic
/// dispatch is the right tool: the gate holds a `Vec<Box<dyn ValidationCheck>>`
/// and adding a gate means adding one value to that list.
pub trait ValidationCheck {
    /// Stable check name, as printed by the gate.
    fn name(&self) -> &'static str;

    /// Measure the property.
    fn run(&self, ctx: &ValidationContext<'_>) -> Result<Evidence>;
}

/// Outcome of a single check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum CheckStatus {
    /// The check held.
    Pass,
    /// The check failed; the gate must fail with it.
    Fail,
}

impl CheckStatus {
    /// Whether this status is a pass.
    #[inline]
    pub fn is_pass(self) -> bool {
        matches!(self, CheckStatus::Pass)
    }

    /// Build a status from a boolean condition.
    #[inline]
    pub fn from_bool(ok: bool) -> Self {
        if ok {
            CheckStatus::Pass
        } else {
            CheckStatus::Fail
        }
    }
}

/// A named check with the evidence behind it.
#[derive(Debug, Clone, Serialize)]
pub struct Check {
    /// Short check name, as printed by the gate.
    pub name: String,
    /// Outcome.
    pub status: CheckStatus,
    /// Human-readable evidence, including the measured numbers.
    pub detail: String,
}

impl Check {
    /// Build a check from a condition and its evidence.
    pub fn new(name: &str, ok: bool, detail: impl Into<String>) -> Self {
        Self {
            name: name.to_string(),
            status: CheckStatus::from_bool(ok),
            detail: detail.into(),
        }
    }

    /// Label a check's evidence with its name.
    pub fn from_evidence(name: &str, evidence: Evidence) -> Self {
        Self::new(name, evidence.passed, evidence.detail)
    }

    /// Whether the check passed.
    #[inline]
    pub fn passed(&self) -> bool {
        self.status.is_pass()
    }
}

pub use gate::{GateReport, canonical_checks, run_gate};
