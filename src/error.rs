//! Error types for the market-information-capacity library.

use std::path::PathBuf;

/// Errors produced by configuration, simulation and output routines.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A configuration value violates a modelling invariant.
    #[error("invalid configuration: {0}")]
    Config(String),

    /// A configuration file could not be read.
    #[error("could not read config file {path}: {source}")]
    ConfigRead {
        /// Path that failed to load.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// A configuration file could not be parsed as TOML.
    #[error("could not parse config file {path}: {source}")]
    ConfigParse {
        /// Path that failed to parse.
        path: PathBuf,
        /// Underlying TOML error.
        #[source]
        source: toml::de::Error,
    },

    /// An output file could not be written.
    #[error("could not write {path}: {source}")]
    Write {
        /// Path that failed to write.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// A CSV record could not be serialised.
    #[error("csv serialisation failed: {0}")]
    Csv(#[from] csv::Error),

    /// A JSON document could not be serialised.
    #[error("json serialisation failed: {0}")]
    Json(#[from] serde_json::Error),

    /// A validation gate failed; the caller should exit non-zero.
    #[error("validation gate failed: {0}")]
    GateFailed(String),

    /// An experiment was asked for something structurally impossible.
    #[error("experiment error: {0}")]
    Experiment(String),
}

/// Convenience result alias.
pub type Result<T> = std::result::Result<T, Error>;
