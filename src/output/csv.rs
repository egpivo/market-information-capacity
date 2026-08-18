//! CSV output.
//!
//! Result files are the boundary between Rust and everything downstream. They
//! are tidy, one row per parameter cell, and always carry the analytic
//! reference value next to the simulated one.

use std::path::Path;

use serde::Serialize;

use crate::error::{Error, Result};

/// Write rows to `path`, creating parent directories as needed.
pub fn write_rows<T: Serialize>(path: impl AsRef<Path>, rows: &[T]) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|source| Error::Write {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let mut writer = csv::Writer::from_path(path)?;
    for row in rows {
        writer.serialize(row)?;
    }
    writer.flush().map_err(|source| Error::Write {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize)]
    struct Row {
        k: usize,
        floor: f64,
    }

    #[test]
    fn writes_a_header_and_rows() {
        let dir = std::env::temp_dir().join("mic-csv-test");
        let path = dir.join("rows.csv");
        write_rows(&path, &[Row { k: 2, floor: 0.75 }]).expect("write");
        let text = std::fs::read_to_string(&path).expect("read");
        assert!(text.starts_with("k,floor\n"));
        assert!(text.contains("2,0.75"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
