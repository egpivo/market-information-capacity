//! JSON summary output.
//!
//! CSVs carry the cell-level results; the JSON summary carries the run-level
//! provenance a reader needs to reproduce them — master seed, configuration,
//! realization counts and gate outcomes.

use std::path::Path;

use serde::Serialize;

use crate::error::{Error, Result};

/// Write a pretty-printed JSON document to `path`.
pub fn write_json<T: Serialize>(path: impl AsRef<Path>, value: &T) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|source| Error::Write {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let text = serde_json::to_string_pretty(value)?;
    std::fs::write(path, text + "\n").map_err(|source| Error::Write {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn writes_pretty_json() {
        let dir = std::env::temp_dir().join("mic-json-test");
        let path = dir.join("summary.json");
        write_json(&path, &json!({"gate": "PASS"})).expect("write");
        let text = std::fs::read_to_string(&path).expect("read");
        assert!(text.contains("\"gate\": \"PASS\""));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
