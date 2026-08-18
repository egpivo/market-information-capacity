//! Typed configuration for the model and the canonical experiments.
//!
//! Every number that defines an experiment lives here or in a TOML file that
//! deserialises into these types. The three canonical worlds are *parameter
//! regimes*, not three separate algorithms: they appear as entries in
//! [`WorldsSpec::presets`] and are executed by the same simulator.
//!
//! All sections carry `#[serde(default)]`, so a TOML file only needs to state
//! what it changes; anything omitted falls back to the canonical v4 value.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::model::traders::{InterpretationNoise, Representation};
use crate::rng::MASTER_SEED;

/// The fundamental source layer: how many independent draws exist and how
/// correlated their errors are.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SourceConfig {
    /// Source budget `K`: the number of fundamental source draws.
    pub count: usize,
    /// Source error scale `sigma_s`.
    pub sigma: f64,
    /// Correlation `rho_s` across source errors.
    pub correlation: f64,
}

/// The trader layer: how many participants read those sources, and how noisily.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TraderConfig {
    /// Trader count `N_T`.
    pub count: usize,
    /// Idiosyncratic interpretation noise scale `sigma_nu`.
    pub interpretation_sigma: f64,
    /// Population clientele / participation tilt `b`.
    pub clientele_bias: f64,
    /// How traders are spread across sources.
    #[serde(default)]
    pub representation: Representation,
    /// How the averaged interpretation error is realized.
    #[serde(default)]
    pub noise: InterpretationNoise,
}

/// The official signal arriving after the pre-official window.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OfficialSignalConfig {
    /// Official signal error scale `sigma_o`.
    pub sigma: f64,
    /// Fixed revision weight `w` placed on the official signal.
    pub weight: f64,
}

impl Default for OfficialSignalConfig {
    fn default() -> Self {
        Self {
            sigma: 0.35,
            weight: 0.8,
        }
    }
}

/// A complete market: sources, traders and the official signal.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MarketConfig {
    /// Fundamental source layer.
    pub sources: SourceConfig,
    /// Trader layer.
    pub traders: TraderConfig,
    /// Official signal.
    #[serde(default)]
    pub official: OfficialSignalConfig,
}

impl MarketConfig {
    /// Check every layer's invariants.
    pub fn validate(&self) -> Result<()> {
        self.sources.validate()?;
        self.traders.validate()?;
        self.official.validate()
    }
}

/// A named parameter regime ("world").
///
/// Worlds are flat in TOML because they are read and edited by hand.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorldSpec {
    /// Display name, e.g. `"Fast Follower"`.
    pub name: String,
    /// Trader count `N_T`.
    pub traders: usize,
    /// Source budget `K`.
    pub sources: usize,
    /// Source correlation `rho_s`.
    pub rho_s: f64,
    /// Source error scale `sigma_s`.
    pub sigma_s: f64,
    /// Clientele tilt `b`.
    pub clientele_bias: f64,
    /// Interpretation noise scale `sigma_nu`.
    pub interpretation_sigma: f64,
}

impl WorldSpec {
    /// Build the market configuration for this world.
    pub fn market(&self, official: OfficialSignalConfig) -> MarketConfig {
        MarketConfig {
            sources: SourceConfig {
                count: self.sources,
                sigma: self.sigma_s,
                correlation: self.rho_s,
            },
            traders: TraderConfig {
                count: self.traders,
                interpretation_sigma: self.interpretation_sigma,
                clientele_bias: self.clientele_bias,
                representation: Representation::Equal,
                noise: InterpretationNoise::Aggregated,
            },
            official,
        }
    }
}

/// The finite-source asymptote benchmark.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AsymptoteSpec {
    /// Source budgets to benchmark.
    pub k_values: Vec<usize>,
    /// Source correlations to benchmark.
    pub rho_values: Vec<f64>,
    /// Trader counts to benchmark.
    pub trader_values: Vec<usize>,
    /// Source error scale.
    pub sigma_s: f64,
    /// Interpretation noise scale (kept small so the floor dominates).
    pub interpretation_sigma: f64,
    /// Clientele tilt (zero for the clean floor benchmark).
    pub clientele_bias: f64,
    /// Realizations per cell.
    pub realizations: usize,
}

impl Default for AsymptoteSpec {
    fn default() -> Self {
        Self {
            k_values: vec![2, 10, 50],
            rho_values: vec![0.0, 0.5, 0.9],
            trader_values: vec![1000, 5000],
            sigma_s: 1.0,
            interpretation_sigma: 0.15,
            clientele_bias: 0.0,
            realizations: 100_000,
        }
    }
}

/// The traders-versus-sources grid.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GridSpec {
    /// Trader counts on the grid.
    pub trader_values: Vec<usize>,
    /// Source budgets on the grid.
    pub k_values: Vec<usize>,
    /// Source correlation held fixed across the grid.
    pub rho_s: f64,
    /// Source error scale held fixed across the grid.
    pub sigma_s: f64,
    /// Interpretation noise scale.
    pub interpretation_sigma: f64,
    /// Clientele tilt.
    pub clientele_bias: f64,
    /// Realizations per seed block.
    pub realizations_per_block: usize,
    /// Number of independent seed blocks per cell.
    pub blocks: usize,
    /// Relative gaps used for convergence thresholds.
    pub convergence_gaps: Vec<f64>,
    /// Source budgets reported in the convergence-threshold table.
    pub convergence_k_values: Vec<usize>,
}

impl Default for GridSpec {
    fn default() -> Self {
        Self {
            trader_values: vec![50, 100, 250, 500, 1000, 2500, 5000],
            k_values: vec![1, 2, 5, 10, 25, 50, 100],
            rho_s: 0.5,
            sigma_s: 1.0,
            interpretation_sigma: 0.8,
            clientele_bias: 0.0,
            realizations_per_block: 5_000,
            blocks: 3,
            convergence_gaps: vec![0.05, 0.01],
            convergence_k_values: vec![2, 10, 50],
        }
    }
}

/// One density point of the doubling comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DensitySpec {
    /// Label reported in the CSV.
    pub label: String,
    /// Base trader count.
    pub traders: usize,
}

/// Doubling traders versus doubling sources.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DoublingSpec {
    /// Base source budget.
    pub k_sources: usize,
    /// Source correlation.
    pub rho_s: f64,
    /// Source error scale.
    pub sigma_s: f64,
    /// Interpretation noise scale.
    pub interpretation_sigma: f64,
    /// Realizations per arm.
    pub realizations: usize,
    /// Density points to compare.
    pub densities: Vec<DensitySpec>,
}

impl Default for DoublingSpec {
    fn default() -> Self {
        Self {
            k_sources: 10,
            rho_s: 0.5,
            sigma_s: 1.0,
            interpretation_sigma: 0.8,
            realizations: 100_000,
            densities: vec![
                DensitySpec {
                    label: "low".into(),
                    traders: 100,
                },
                DensitySpec {
                    label: "medium".into(),
                    traders: 1000,
                },
                DensitySpec {
                    label: "high".into(),
                    traders: 5000,
                },
            ],
        }
    }
}

/// The three-worlds experiment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WorldsSpec {
    /// Realizations per world.
    pub realizations: usize,
    /// Cap on retained absolute revisions used for decile reporting.
    pub revision_samples: usize,
    /// The canonical parameter regimes.
    pub presets: Vec<WorldSpec>,
}

impl Default for WorldsSpec {
    fn default() -> Self {
        Self {
            realizations: 30_000,
            revision_samples: 200_000,
            presets: canonical_worlds(),
        }
    }
}

/// The canonical v4 worlds.
pub fn canonical_worlds() -> Vec<WorldSpec> {
    vec![
        WorldSpec {
            name: "Informative".into(),
            traders: 2500,
            sources: 50,
            rho_s: 0.1,
            sigma_s: 0.45,
            clientele_bias: 0.05,
            interpretation_sigma: 0.8,
        },
        WorldSpec {
            name: "Selected Clientele".into(),
            traders: 2500,
            sources: 10,
            rho_s: 0.5,
            sigma_s: 0.75,
            clientele_bias: 0.35,
            interpretation_sigma: 0.8,
        },
        WorldSpec {
            name: "Fast Follower".into(),
            traders: 2500,
            sources: 2,
            rho_s: 0.9,
            sigma_s: 1.0,
            clientele_bias: 0.35,
            interpretation_sigma: 0.8,
        },
    ]
}

/// The same-price conditional experiment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SamePriceSpec {
    /// Half-width of the accepted price band, `|P_OC| < band`.
    pub band: f64,
    /// Realizations drawn per world before conditioning.
    pub realizations: usize,
    /// Cap on retained accepted latent values.
    pub retained_samples: usize,
}

impl Default for SamePriceSpec {
    fn default() -> Self {
        Self {
            band: 0.15,
            realizations: 300_000,
            retained_samples: 400_000,
        }
    }
}

/// One-factor-at-a-time sweeps around a baseline regime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SweepSpec {
    /// Source error scale values.
    pub sigma_s: Vec<f64>,
    /// Source correlation values.
    pub rho_s: Vec<f64>,
    /// Source budget values.
    pub sources: Vec<usize>,
    /// Interpretation noise values.
    pub interpretation_sigma: Vec<f64>,
    /// Trader count values.
    pub traders: Vec<usize>,
    /// Clientele tilt values.
    pub clientele_bias: Vec<f64>,
}

impl Default for SweepSpec {
    fn default() -> Self {
        Self {
            sigma_s: vec![0.35, 0.5, 0.75, 1.0, 1.4, 1.8],
            rho_s: vec![0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
            sources: vec![1, 2, 5, 10, 25, 50, 100],
            interpretation_sigma: vec![0.0, 0.2, 0.4, 0.8, 1.6],
            traders: vec![50, 100, 250, 500, 1000, 2500, 5000],
            clientele_bias: vec![0.0, 0.1, 0.35, 0.7, 1.2],
        }
    }
}

/// One labelled slice of the information-independence phase diagram.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhaseSliceSpec {
    /// Slice label.
    pub label: String,
    /// Source budget.
    pub sources: usize,
    /// Source correlation.
    pub rho_s: f64,
}

/// The phase-slice sweep over source precision and clientele tilt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PhaseSpec {
    /// Trader count held fixed across slices.
    pub traders: usize,
    /// Interpretation noise held fixed across slices.
    pub interpretation_sigma: f64,
    /// Geometric grid of source precision: lower bound.
    pub sigma_s_min: f64,
    /// Geometric grid of source precision: upper bound.
    pub sigma_s_max: f64,
    /// Number of source-precision steps.
    pub sigma_s_steps: usize,
    /// Linear grid of clientele tilt: lower bound.
    pub bias_min: f64,
    /// Linear grid of clientele tilt: upper bound.
    pub bias_max: f64,
    /// Number of clientele-tilt steps.
    pub bias_steps: usize,
    /// Realizations per block.
    pub realizations_per_block: usize,
    /// Number of seed blocks.
    pub blocks: usize,
    /// The slices to sweep.
    pub slices: Vec<PhaseSliceSpec>,
}

impl Default for PhaseSpec {
    fn default() -> Self {
        Self {
            traders: 2500,
            interpretation_sigma: 0.8,
            sigma_s_min: 0.35,
            sigma_s_max: 1.8,
            sigma_s_steps: 9,
            bias_min: 0.0,
            bias_max: 1.5,
            bias_steps: 9,
            realizations_per_block: 5_000,
            blocks: 3,
            slices: vec![
                PhaseSliceSpec {
                    label: "rich".into(),
                    sources: 100,
                    rho_s: 0.1,
                },
                PhaseSliceSpec {
                    label: "moderate".into(),
                    sources: 10,
                    rho_s: 0.5,
                },
                PhaseSliceSpec {
                    label: "thin".into(),
                    sources: 2,
                    rho_s: 0.9,
                },
            ],
        }
    }
}

/// The sensitivity experiment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SensitivitySpec {
    /// Realizations per block for the one-factor sweeps.
    pub realizations_per_block: usize,
    /// Number of seed blocks per sweep point.
    pub blocks: usize,
    /// Baseline regime the sweeps perturb.
    pub baseline: WorldSpec,
    /// The one-factor sweeps.
    pub sweeps: SweepSpec,
    /// The phase-slice sweep.
    pub phase: PhaseSpec,
}

impl Default for SensitivitySpec {
    fn default() -> Self {
        Self {
            realizations_per_block: 20_000,
            blocks: 3,
            baseline: WorldSpec {
                name: "baseline".into(),
                traders: 2500,
                sources: 10,
                rho_s: 0.5,
                sigma_s: 0.75,
                clientele_bias: 0.35,
                interpretation_sigma: 0.8,
            },
            sweeps: SweepSpec::default(),
            phase: PhaseSpec::default(),
        }
    }
}

/// Scale of the validation gate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ValidationSpec {
    /// Realizations per Monte Carlo cell used by the gate.
    pub realizations: usize,
    /// Realizations drawn for the conditional same-price gate.
    pub same_price_realizations: usize,
    /// Trader counts used by the plateau and duplication gates.
    pub plateau_trader_values: Vec<usize>,
    /// Tolerance in standard errors for analytic/Monte Carlo parity.
    pub parity_z_tolerance: f64,
}

impl Default for ValidationSpec {
    fn default() -> Self {
        Self {
            realizations: 200_000,
            same_price_realizations: 400_000,
            plateau_trader_values: vec![100, 1_000, 10_000, 100_000],
            parity_z_tolerance: 4.0,
        }
    }
}

/// The complete experiment configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Master seed for every derived stream.
    pub master_seed: u64,
    /// Number of deterministic parallel batches per Monte Carlo block.
    pub batches: usize,
    /// Official signal shared by every world.
    pub official: OfficialSignalConfig,
    /// Finite-source asymptote benchmark.
    pub asymptote: AsymptoteSpec,
    /// Traders-versus-sources grid.
    pub grid: GridSpec,
    /// Doubling comparison.
    pub doubling: DoublingSpec,
    /// Three-worlds experiment.
    pub worlds: WorldsSpec,
    /// Same-price conditional experiment.
    pub same_price: SamePriceSpec,
    /// Sensitivity experiment.
    pub sensitivity: SensitivitySpec,
    /// Validation gate scale.
    pub validation: ValidationSpec,
}

impl Default for Config {
    fn default() -> Self {
        Self::publication()
    }
}

impl Config {
    /// The canonical publication-scale configuration.
    pub fn publication() -> Self {
        Self {
            master_seed: MASTER_SEED,
            batches: 16,
            official: OfficialSignalConfig::default(),
            asymptote: AsymptoteSpec::default(),
            grid: GridSpec::default(),
            doubling: DoublingSpec::default(),
            worlds: WorldsSpec::default(),
            same_price: SamePriceSpec::default(),
            sensitivity: SensitivitySpec::default(),
            validation: ValidationSpec::default(),
        }
    }

    /// A reduced-scale configuration for development and integration tests.
    ///
    /// Parameter grids are unchanged; only realization counts shrink, so the
    /// qualitative structure of every result is preserved while runtimes drop
    /// by roughly two orders of magnitude.
    pub fn quick() -> Self {
        let mut cfg = Self::publication();
        cfg.batches = 8;
        cfg.asymptote.realizations = 20_000;
        cfg.grid.realizations_per_block = 1_000;
        cfg.doubling.realizations = 20_000;
        cfg.worlds.realizations = 10_000;
        cfg.same_price.realizations = 60_000;
        cfg.sensitivity.realizations_per_block = 4_000;
        cfg.sensitivity.phase.realizations_per_block = 1_000;
        cfg.validation.realizations = 50_000;
        cfg.validation.same_price_realizations = 100_000;
        cfg.validation.plateau_trader_values = vec![100, 1_000, 10_000];
        cfg
    }

    /// Load a configuration from a TOML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| Error::ConfigRead {
            path: path.to_path_buf(),
            source,
        })?;
        let cfg: Self = toml::from_str(&text).map_err(|source| Error::ConfigParse {
            path: path.to_path_buf(),
            source,
        })?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Check every experiment section for structurally impossible settings.
    pub fn validate(&self) -> Result<()> {
        self.official.validate()?;
        if self.batches == 0 {
            return Err(Error::Config("batches must be at least 1".into()));
        }
        if self.asymptote.realizations < 2 {
            return Err(Error::Config(
                "asymptote realizations must be at least 2".into(),
            ));
        }
        if self.asymptote.k_values.is_empty()
            || self.asymptote.rho_values.is_empty()
            || self.asymptote.trader_values.is_empty()
        {
            return Err(Error::Config("asymptote grid must not be empty".into()));
        }
        if self.grid.blocks < 2 {
            return Err(Error::Config(
                "grid needs at least two seed blocks for a block standard error".into(),
            ));
        }
        if self.grid.trader_values.is_empty() || self.grid.k_values.is_empty() {
            return Err(Error::Config("grid must not be empty".into()));
        }
        if self.doubling.densities.is_empty() {
            return Err(Error::Config(
                "doubling comparison needs at least one density".into(),
            ));
        }
        if self.worlds.presets.is_empty() {
            return Err(Error::Config("at least one world must be defined".into()));
        }
        if self.same_price.band <= 0.0 {
            return Err(Error::Config("same-price band must be positive".into()));
        }
        if self.sensitivity.blocks < 2 {
            return Err(Error::Config(
                "sensitivity needs at least two seed blocks".into(),
            ));
        }
        for world in &self.worlds.presets {
            world.market(self.official).validate()?;
        }
        self.sensitivity.baseline.market(self.official).validate()?;
        if self.validation.realizations < 2 || self.validation.plateau_trader_values.is_empty() {
            return Err(Error::Config(
                "validation gate needs realizations and at least one trader count".into(),
            ));
        }
        Ok(())
    }

    /// Number of realizations split across batches, as a per-batch schedule.
    ///
    /// The schedule is fixed before any thread starts, so batch `b` always
    /// covers the same count regardless of scheduling.
    pub fn batch_schedule(&self, realizations: usize) -> Vec<usize> {
        batch_schedule(realizations, self.batches)
    }
}

/// Split `realizations` into at most `batches` deterministic chunks.
pub fn batch_schedule(realizations: usize, batches: usize) -> Vec<usize> {
    if realizations == 0 {
        return Vec::new();
    }
    let batches = batches.max(1).min(realizations);
    let base = realizations / batches;
    let remainder = realizations % batches;
    (0..batches)
        .map(|b| base + usize::from(b < remainder))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publication_config_validates() {
        Config::publication().validate().expect("canonical config");
        Config::quick().validate().expect("quick config");
    }

    #[test]
    fn canonical_worlds_match_v4() {
        let worlds = canonical_worlds();
        assert_eq!(worlds.len(), 3);
        let ff = &worlds[2];
        assert_eq!(ff.name, "Fast Follower");
        assert_eq!(ff.traders, 2500);
        assert_eq!(ff.sources, 2);
        assert!((ff.rho_s - 0.9).abs() < 1e-12);
        assert!((ff.sigma_s - 1.0).abs() < 1e-12);
        assert!((ff.clientele_bias - 0.35).abs() < 1e-12);
    }

    #[test]
    fn batch_schedule_conserves_realizations() {
        for (n, b) in [(100usize, 7usize), (5, 16), (0, 4), (1_000_000, 16)] {
            let schedule = batch_schedule(n, b);
            assert_eq!(schedule.iter().sum::<usize>(), n);
            assert!(schedule.iter().all(|&c| c > 0) || n == 0);
        }
    }

    #[test]
    fn partial_toml_falls_back_to_canonical_values() {
        let cfg: Config = toml::from_str("master_seed = 7\n").expect("parse");
        assert_eq!(cfg.master_seed, 7);
        assert_eq!(cfg.grid.k_values, GridSpec::default().k_values);
        assert_eq!(cfg.worlds.presets.len(), 3);
    }

    #[test]
    fn invalid_world_is_rejected() {
        let mut cfg = Config::publication();
        cfg.worlds.presets[0].rho_s = 1.4;
        assert!(cfg.validate().is_err());
    }
}
