//! The information hierarchy `V -> s_k -> m_i -> P_OC`.
//!
//! Each layer is a separate module and a separate type:
//!
//! * [`latent`] draws the latent valuation `V`;
//! * [`sources`] draws the `K` fundamental sources — the only place new
//!   information about `V` enters the model;
//! * [`traders`] maps existing sources onto `N_T` trader beliefs;
//! * [`market`] clears those beliefs into the onchain price `P_OC`;
//! * [`official`] revises the price when external information arrives.
//!
//! The separation is load bearing: because [`traders::TraderLayer`] cannot
//! reach the source generator, no amount of trader growth can manufacture
//! fundamental information.

pub mod latent;
pub mod market;
pub mod official;
pub mod sources;
pub mod traders;

pub use latent::Latent;
pub use market::{MarketSimulator, Realization, RevisedRealization, Workspace};
pub use sources::{SourceDraw, SourceLayer};
pub use traders::{InterpretationNoise, Representation, TraderLayer};
