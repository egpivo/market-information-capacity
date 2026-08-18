//! The information hierarchy `V -> s_k -> m_i -> P_OC`.
//!
//! # Architecture
//!
//! ```text
//! LatentProcess
//!       |  LatentValue
//!       v
//! InformationSourceModel
//!       |  SourceSet
//!       v
//! BeliefFormation
//!       |  BeliefSet
//!       v
//! ClearingRule
//!       |  MarketPrice
//!       v
//! RevisionRule  <--  ExternalSignal  <--  ExternalSignalProcess
//! ```
//!
//! Each arrow is a trait, and each trait is one economic degree of freedom:
//! what the market is trying to price, how much can independently be known
//! about it, how participants interpret what is known, how those views become a
//! price, what new information arrives from outside, and how the market
//! assimilates it.
//!
//! # The dependency rule
//!
//! **Fundamental information may only be created by the source layer.**
//!
//! * [`BeliefFormation::form`](traders::BeliefFormation::form) receives a
//!   [`SourceSet`], not a [`LatentValue`].
//! * [`ClearingRule::clear`](clearing::ClearingRule::clear) receives a
//!   [`BeliefSet`], not sources and not latent state.
//! * [`RevisionRule::revise`](official::RevisionRule::revise) receives a
//!   [`MarketPrice`] and an [`ExternalSignal`], not latent state.
//!
//! This direction is deliberate. It prevents a trader count from implicitly
//! increasing the market's fundamental information capacity — the failure of the
//! superseded v2 model — and it prevents a clearing or revision rule from
//! quietly becoming clairvoyant. Because the restriction lives in the
//! signatures, a future implementation cannot violate it without changing a
//! trait, which is a visible and reviewable act.
//!
//! # Composition
//!
//! [`MarketEngine`](market::MarketEngine) wires the six components together with
//! static dispatch, so there is no vtable in the Monte Carlo hot loop.
//! [`CanonicalMarket`](market::CanonicalMarket) names the v4 composition that
//! produces every canonical result.

pub mod clearing;
pub mod latent;
pub mod market;
pub mod official;
pub mod sources;
pub mod traders;
pub mod types;

pub use clearing::{ClearingRule, EqualRepresentationClearing};
pub use latent::{GaussianLatent, LatentProcess};
pub use market::{CanonicalMarket, MarketEngine, Realization, RevisedRealization, Workspace};
pub use official::{
    ExternalSignalProcess, FixedWeightRevision, OfficialSignalProcess, RevisionRule,
};
pub use sources::{CorrelatedFiniteSources, InformationSourceModel};
pub use traders::{BeliefFormation, InterpretationNoise, Representation, SourceAttachedBeliefs};
pub use types::{
    BeliefSet, ExternalSignal, FundamentalSignal, LatentValue, MarketPrice, SourceSet, TraderBelief,
};
