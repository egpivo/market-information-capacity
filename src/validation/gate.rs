//! The composite validation gate.
//!
//! Runs every check, renders the report, and reports a single pass/fail that
//! the CLI turns into an exit code.

use serde::Serialize;

use crate::config::Config;
use crate::error::Result;
use crate::validation::{
    Check, ValidationCheck, ValidationContext, analytic_floor, determinism, fast_follower,
    finite_source, hierarchy, same_price,
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

/// The canonical gate, in report order.
///
/// Adding a gate means adding one value to this list. The report order is the
/// list order, and it runs from structural checks through analytic identities to
/// economic mechanisms.
pub fn canonical_checks() -> Vec<Box<dyn ValidationCheck>> {
    vec![
        Box::new(hierarchy::HierarchyIntact),
        Box::new(hierarchy::TraderDuplicationIsNotInformation),
        Box::new(analytic_floor::AnalyticFloorIdentities),
        Box::new(finite_source::MonteCarloFloorParity),
        Box::new(finite_source::FiniteSourcePlateau),
        Box::new(finite_source::SourceCountComparativeStatic),
        Box::new(fast_follower::FastFollowerMechanism),
        Box::new(same_price::SamePriceConditional),
        Box::new(determinism::DeterministicReproduction),
    ]
}

/// Run every validation check.
pub fn run_gate(cfg: &Config) -> Result<GateReport> {
    run_checks(cfg, &canonical_checks())
}

/// Run a specific list of checks.
pub fn run_checks(cfg: &Config, checks: &[Box<dyn ValidationCheck>]) -> Result<GateReport> {
    let ctx = ValidationContext::new(cfg);
    let checks = checks
        .iter()
        .map(|check| {
            check
                .run(&ctx)
                .map(|evidence| Check::from_evidence(check.name(), evidence))
        })
        .collect::<Result<Vec<_>>>()?;
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
    use crate::validation::{CheckStatus, Evidence};

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

    #[test]
    fn the_canonical_gate_lists_every_check_once() {
        let checks = canonical_checks();
        assert_eq!(checks.len(), 9);
        let mut names: Vec<&str> = checks.iter().map(|c| c.name()).collect();
        let count = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), count, "duplicate check names in the gate");
    }

    /// A gate that cannot fail is not a gate: a check reporting failure must
    /// propagate to the report.
    #[test]
    fn a_failing_check_fails_the_report() {
        struct AlwaysFails;
        impl ValidationCheck for AlwaysFails {
            fn name(&self) -> &'static str {
                "always fails"
            }
            fn run(&self, _ctx: &ValidationContext<'_>) -> Result<Evidence> {
                Ok(Evidence::new(false, "by construction"))
            }
        }
        let cfg = Config::quick();
        let checks: Vec<Box<dyn ValidationCheck>> = vec![Box::new(AlwaysFails)];
        let report = run_checks(&cfg, &checks).expect("runs");
        assert!(!report.passed);
        assert!(report.render().contains("MODEL_GATE: FAIL"));
    }
}
