//! Market clearing: turning trader beliefs into the observable onchain price.
//!
//! The executable price is the market-clearing aggregation of trader beliefs.
//! With identical risk tolerance across traders the clearing weights are
//! uniform, so
//!
//! ```text
//! P_OC = (1 / N_T) * sum_i m_i
//!      = sum_k w_k * s_k + b + nu_bar,
//! ```
//!
//! where `w_k` are the representation weights from the trader layer, `b` is the
//! population clientele tilt, and `nu_bar` is the averaged interpretation
//! error. The second line is the identity that makes the information floor
//! visible: the only `V`-bearing term is the source aggregate, whose precision
//! is set by `K` and `rho_s` alone.

use rand::Rng;

use crate::config::{MarketConfig, OfficialSignalConfig};
use crate::error::Result;
use crate::model::latent::{Latent, draw_latent};
use crate::model::official::{draw_official, revise};
use crate::model::sources::{SourceDraw, SourceLayer};
use crate::model::traders::TraderLayer;

/// One pre-official realization of the market.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Realization {
    /// The latent valuation `V`.
    pub latent: f64,
    /// The pre-official onchain price `P_OC`.
    pub pre_price: f64,
}

impl Realization {
    /// Squared pricing error against the latent value.
    #[inline]
    pub fn squared_error(&self) -> f64 {
        let e = self.pre_price - self.latent;
        e * e
    }
}

/// One realization extended through the official-information boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RevisedRealization {
    /// The pre-official realization.
    pub pre: Realization,
    /// The official signal about `V`.
    pub official: f64,
    /// The post-boundary price.
    pub post_price: f64,
}

impl RevisedRealization {
    /// Squared post-boundary pricing error.
    #[inline]
    pub fn post_squared_error(&self) -> f64 {
        let e = self.post_price - self.pre.latent;
        e * e
    }

    /// Absolute price revision across the boundary.
    #[inline]
    pub fn abs_revision(&self) -> f64 {
        (self.post_price - self.pre.pre_price).abs()
    }
}

/// Reusable per-thread scratch space, so the hot loop does not allocate.
#[derive(Debug, Clone, Default)]
pub struct Workspace {
    sources: SourceDraw,
}

/// A configured market: latent value, source layer, trader layer, official signal.
#[derive(Debug, Clone)]
pub struct MarketSimulator {
    sources: SourceLayer,
    traders: TraderLayer,
    official: OfficialSignalConfig,
}

impl MarketSimulator {
    /// Build a simulator from a market configuration.
    pub fn new(cfg: &MarketConfig) -> Result<Self> {
        cfg.validate()?;
        Ok(Self {
            sources: SourceLayer::new(&cfg.sources)?,
            traders: TraderLayer::new(&cfg.traders)?,
            official: cfg.official,
        })
    }

    /// Allocate scratch space sized for this market's source budget.
    pub fn workspace(&self) -> Workspace {
        Workspace {
            sources: SourceDraw::with_capacity(self.sources.count()),
        }
    }

    /// The fundamental source layer.
    #[inline]
    pub fn sources(&self) -> &SourceLayer {
        &self.sources
    }

    /// The trader layer.
    #[inline]
    pub fn traders(&self) -> &TraderLayer {
        &self.traders
    }

    /// Draw one pre-official realization.
    ///
    /// Draw order is fixed as `V`, common source error, source-specific errors,
    /// interpretation noise, which keeps a stream's output stable across
    /// refactors of the calling code.
    pub fn realize<R: Rng + ?Sized>(&self, ws: &mut Workspace, rng: &mut R) -> Realization {
        let latent = draw_latent(rng);
        self.sources.draw_into(latent, rng, &mut ws.sources);
        let pre_price = self.traders.mean_belief(&ws.sources, rng);
        Realization {
            latent: latent.value(),
            pre_price,
        }
    }

    /// Extend a realization through the official-information boundary.
    pub fn revise_with_official<R: Rng + ?Sized>(
        &self,
        pre: Realization,
        rng: &mut R,
    ) -> RevisedRealization {
        let official = draw_official(Latent(pre.latent), &self.official, rng);
        let post_price = revise(pre.pre_price, official, self.official.weight);
        RevisedRealization {
            pre,
            official,
            post_price,
        }
    }

    /// Draw a realization and immediately revise it.
    pub fn realize_revised<R: Rng + ?Sized>(
        &self,
        ws: &mut Workspace,
        rng: &mut R,
    ) -> RevisedRealization {
        let pre = self.realize(ws, rng);
        self.revise_with_official(pre, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{SourceConfig, TraderConfig};
    use crate::model::traders::{InterpretationNoise, Representation};
    use crate::rng::{MASTER_SEED, StreamId, stream};

    fn market(k: usize, n: usize, interpretation_sigma: f64) -> MarketConfig {
        MarketConfig {
            sources: SourceConfig {
                count: k,
                sigma: 1.0,
                correlation: 0.5,
            },
            traders: TraderConfig {
                count: n,
                interpretation_sigma,
                clientele_bias: 0.0,
                representation: Representation::Equal,
                noise: InterpretationNoise::Aggregated,
            },
            official: OfficialSignalConfig {
                sigma: 0.35,
                weight: 0.8,
            },
        }
    }

    /// With interpretation noise switched off, the trader count cannot move the
    /// price at all: the price is a deterministic function of the source layer.
    #[test]
    fn trader_count_cannot_change_a_noiseless_price() {
        let small = MarketSimulator::new(&market(4, 100, 0.0)).expect("valid");
        let large = MarketSimulator::new(&market(4, 1_000_000, 0.0)).expect("valid");
        let mut ws_s = small.workspace();
        let mut ws_l = large.workspace();
        let mut rng_s = stream(MASTER_SEED, StreamId::new("test-market", 0, 0, 0));
        let mut rng_l = stream(MASTER_SEED, StreamId::new("test-market", 0, 0, 0));
        for _ in 0..1000 {
            let a = small.realize(&mut ws_s, &mut rng_s);
            let b = large.realize(&mut ws_l, &mut rng_l);
            assert_eq!(a.latent, b.latent);
            assert_eq!(a.pre_price, b.pre_price);
        }
    }
}
