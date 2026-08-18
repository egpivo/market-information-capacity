//! The composite validation gate.
//!
//! Runs every check, renders the report, and reports a single pass/fail that
//! the CLI turns into an exit code.

use serde::Serialize;

use crate::config::Config;
use crate::error::Result;
use crate::validation::{
    Check, analytic_floor, determinism, fast_follower, finite_source, hierarchy, same_price,
};

/// The full gate outcome.
#[derive(Debug, Clone, Serialize)]
pub struct GateReport {
    /// Master seed the gate ran under.
    pub master_seed: u64,
    /// Realizations per Monte Carlo check.
    pub realizations_per_check: usize,
    /// Every check, in report order.
    pub checks: Vec<Check>,
    /// Whether every check passed.
    pub passed: bool,
}

impl GateReport {
    /// Render the report in the canonical fixed-width form.
    pub fn render(&self) -> String {
        let width = self
            .checks
            .iter()
            .map(|c| c.name.len())
            .max()
            .unwrap_or(0)
            .max(24);
        let mut out = String::from("Market Information Capacity — validation\n\n");
        for check in &self.checks {
            let dots = ".".repeat(width + 3 - check.name.len());
            let status = if check.passed() { "PASS" } else { "FAIL" };
            out.push_str(&format!("{} {dots} {status}\n", check.name));
        }
        out.push('\n');
        for check in self.checks.iter().filter(|c| !c.passed()) {
            out.push_str(&format!("  FAILED {}: {}\n", check.name, check.detail));
        }
        out.push_str(&format!(
            "MODEL_GATE: {}\n",
            if self.passed { "PASS" } else { "FAIL" }
        ));
        out
    }

    /// Render the evidence behind every check.
    pub fn render_evidence(&self) -> String {
        self.checks
            .iter()
            .map(|c| format!("  {}: {}", c.name, c.detail))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Run every validation check.
pub fn run_gate(cfg: &Config) -> Result<GateReport> {
    let checks = vec![
        hierarchy::check_hierarchy(cfg)?,
        hierarchy::check_trader_duplication(cfg)?,
        analytic_floor::check_analytic_floor()?,
        finite_source::check_floor_parity(cfg)?,
        finite_source::check_plateau(cfg)?,
        finite_source::check_source_comparative_static(cfg)?,
        fast_follower::check_fast_follower(cfg)?,
        same_price::check_same_price(cfg)?,
        determinism::check_determinism(cfg)?,
    ];
    let passed = checks.iter().all(Check::passed);
    Ok(GateReport {
        master_seed: cfg.master_seed,
        realizations_per_check: cfg.validation.realizations,
        checks,
        passed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::CheckStatus;

    #[test]
    fn render_marks_failures() {
        let report = GateReport {
            master_seed: 1,
            realizations_per_check: 10,
            checks: vec![
                Check::new("alpha", true, "fine"),
                Check::new("beta", false, "measured 3, expected 4"),
            ],
            passed: false,
        };
        let text = report.render();
        assert!(text.contains("alpha"));
        assert!(text.contains("FAIL"));
        assert!(text.contains("measured 3, expected 4"));
        assert!(text.contains("MODEL_GATE: FAIL"));
        assert_eq!(report.checks[1].status, CheckStatus::Fail);
    }
}
