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
//! ## Layout
//!
//! * [`model`] — the information hierarchy, one module per layer;
//! * [`analytics`] — closed-form results and streaming statistics;
//! * [`simulation`] — the deterministic Monte Carlo engine and the canonical
//!   experiments;
//! * [`validation`] — the research-integrity gate;
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
