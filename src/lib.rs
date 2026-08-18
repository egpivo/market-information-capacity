//! # Market Information Capacity
//!
//! More traders do not necessarily mean more information.
//!
//! This crate is the computational source of truth for a single research
//! question: what happens when market participation grows faster than the
//! amount of independent information available to the market?
//!
//! The model is a four-layer information hierarchy,
//!
//! ```text
//! V -> (s_1, ..., s_K) -> (m_1, ..., m_{N_T}) -> P_OC
//! ```
//!
//! in which `N_T`, the trader count, and `K`, the number of fundamental
//! information sources, are **separate state variables**. Traders read existing
//! sources; they do not create new ones. The consequence is an analytic floor on
//! how well the market can price the latent value,
//!
//! ```text
//! MSE_inf(K, rho_s) = sigma_s^2 [ rho_s + (1 - rho_s) / K ],
//! ```
//!
//! in which the trader count does not appear. See
//! [`analytics::information_floor`].
//!
//! ## Architecture
//!
//! Each arrow in the hierarchy is a trait, and each trait is one economic degree
//! of freedom:
//!
//! ```text
//! LatentProcess            what is the market trying to price?
//!       |  LatentValue
//!       v
//! InformationSourceModel   how much can independently be known about it?
//!       |  SourceSet
//!       v
//! BeliefFormation          how do participants interpret what is known?
//!       |  BeliefSet
//!       v
//! ClearingRule             how do those views become a price?
//!       |  MarketPrice
//!       v
//! RevisionRule  <-- ExternalSignal <-- ExternalSignalProcess
//! ```
//!
//! The dependency direction is load bearing.
//! [`BeliefFormation::form`](model::traders::BeliefFormation::form) receives a
//! [`SourceSet`](model::types::SourceSet) and **not** a
//! [`LatentValue`](model::types::LatentValue), so no belief model — present or
//! future — can manufacture fundamental information from trader growth. See
//! [`model`] for the full rule.
//!
//! Composition happens in [`MarketEngine`](model::market::MarketEngine), which is
//! generic over all six components and dispatches statically, so there is no
//! vtable in the Monte Carlo hot loop.
//! [`CanonicalMarket`](model::market::CanonicalMarket) names the composition that
//! produces every canonical result. Dynamic dispatch is used in exactly one
//! place, [`ValidationCheck`](validation::ValidationCheck), where the collection
//! is heterogeneous and enumerated once per run.
//!
//! Metrics, configuration, output formats, world presets and the random number
//! generator are deliberately **not** behind traits. They are data and
//! infrastructure, not economic degrees of freedom, and abstracting them would
//! add indirection without adding a model.
//!
//! ## Layout
//!
//! * [`model`] — the information hierarchy, one trait and one module per layer;
//! * [`analytics`] — closed-form results and streaming statistics;
//! * [`simulation`] — the deterministic Monte Carlo engine and the canonical
//!   experiments, each implementing [`Experiment`](simulation::Experiment);
//! * [`validation`] — the research-integrity gate, a list of
//!   [`ValidationCheck`](validation::ValidationCheck) values;
//! * [`output`] — CSV and JSON serialisation;
//! * [`pipeline`] — command orchestration;
//! * [`config`] — typed configuration, including the three canonical worlds;
//! * [`rng`] — deterministic seed derivation.
//!
//! ## Reproducibility
//!
//! Every canonical number descends from [`rng::MASTER_SEED`] through a pure
//! hash of the work unit's coordinates, so the same configuration and seed
//! reproduce the same results regardless of parallelism.

#![warn(missing_docs)]

pub mod analytics;
pub mod config;
pub mod error;
pub mod model;
pub mod output;
pub mod pipeline;
pub mod rng;
pub mod simulation;
pub mod validation;

pub use config::{Config, MarketConfig, OfficialSignalConfig, SourceConfig, TraderConfig};
pub use error::{Error, Result};
pub use model::{
    BeliefFormation, CanonicalMarket, ClearingRule, ExternalSignalProcess, InformationSourceModel,
    LatentProcess, MarketEngine, RevisionRule,
};
pub use simulation::{Experiment, SimulationContext};
pub use validation::{ValidationCheck, ValidationContext};
