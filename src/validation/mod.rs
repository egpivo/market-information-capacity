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

    /// Whether the check passed.
    #[inline]
    pub fn passed(&self) -> bool {
        self.status.is_pass()
    }
}

pub use gate::{GateReport, run_gate};
