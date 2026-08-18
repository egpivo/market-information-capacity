//! Result serialisation.
//!
//! Rust owns every canonical number; these modules are the only way those
//! numbers leave the process. Downstream tooling — including the Python
//! plotting script — reads these files and nothing else.

pub mod csv;
pub mod json;

pub use csv::write_rows;
pub use json::write_json;
