//! Checkpoints y escrituras atómicas para actualizaciones pequeñas.
//!
//! Cada actualización guarda primero una copia del estado anterior y escribe
//! el nuevo contenido en un archivo temporal. Si la validación posterior falla,
//! el llamador puede revertir al checkpoint con [`Checkpoint::rollback`].

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use crate::error::{Error, Result};

#[derive(Debug, Clone)]
pub struct Checkpoint {
    target: PathBuf,
    backup: Option<PathBuf>,
}

impl Checkpoint {
    pub fn target(&self) -> &Path {
        &self.target
    }

    /// Ruta del checkpoint, o `None` si el archivo todavía no existía.
    pub fn backup(&self) -> Option<&Path> {
        self.backup.as_deref()
    }

    /// Restaura el estado anterior. Si no existía, elimina el archivo nuevo.
    pub fn rollback(&self) -> Result<()> {
        match &self.backup {
            Some(path) => {
                let bytes = fs::read(path).map_err(|e| {
                    Error::new(format!(
                        "no se puede leer el checkpoint {}: {e}",
                        path.display()
                    ))
                })?;
                atomic_write(&self.target, &bytes)
            }
            None => {
                if self.target.exists() {
                    fs::remove_file(&self.target).map_err(|e| {
                        Error::new(format!(
                            "no se puede eliminar {} durante el rollback: {e}",
                            self.target.display()
                        ))
                    })?;
                }
                Ok(())
            }
        }
    }
}

/// Guarda una copia del archivo actual en `dir` antes de modificarlo.
/// El nombre incluye la marca UTC y un contador para no sobrescribir.
pub fn save_checkpoint(target: &Path, dir: &Path, label: &str) -> Result<Checkpoint> {
    fs::create_dir_all(dir).map_err(|e| {
        Error::new(format!(
            "no se puede crear el directorio de checkpoints {}: {e}",
            dir.display()
        ))
    })?;
    let backup = if target.is_file() {
        let name = target
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "archivo".to_string());
        let stamp = utc_stamp();
        let mut candidate = dir.join(format!("{name}.{label}.{stamp}.bak"));
        let mut n = 1usize;
        while candidate.exists() {
            candidate = dir.join(format!("{name}.{label}.{stamp}-{n}.bak"));
            n += 1;
        }
        fs::copy(target, &candidate).map_err(|e| {
            Error::new(format!(
                "no se puede crear el checkpoint {}: {e}",
                candidate.display()
            ))
        })?;
        Some(candidate)
    } else {
        None
    };
    Ok(Checkpoint {
        target: target.to_path_buf(),
        backup,
    })
}

/// Escribe un archivo de forma atómica en la misma carpeta.
pub fn atomic_write(target: &Path, bytes: &[u8]) -> Result<()> {
    let parent = target.parent().filter(|p| !p.as_os_str().is_empty());
    if let Some(parent) = parent {
        fs::create_dir_all(parent)?;
    }
    let name = target
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "archivo".to_string());
    let tmp = target.with_file_name(format!(".{name}.tmp-{}", std::process::id()));
    let result = (|| -> Result<()> {
        let mut file = File::create(&tmp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        Ok(())
    })();
    if let Err(e) = result {
        let _ = fs::remove_file(&tmp);
        return Err(e);
    }
    #[cfg(windows)]
    if target.exists() {
        // Windows no reemplaza con rename; se conserva el checkpoint previo.
        fs::remove_file(target)?;
    }
    fs::rename(&tmp, target).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        Error::new(format!("no se puede reemplazar {}: {e}", target.display()))
    })
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path)
        .map_err(|e| Error::new(format!("no se puede leer {}: {e}", path.display())))?;
    Ok(sha256_bytes(&bytes))
}

/// Marca UTC compacta y ordenable: `YYYYMMDDTHHMMSS.mmmZ`.
pub fn utc_stamp() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let millis = now.subsec_millis();
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}{month:02}{day:02}T{:02}{:02}{:02}.{millis:03}Z",
        rem / 3600,
        (rem / 60) % 60,
        rem % 60
    )
}

/// Días desde 1970-01-01 a fecha civil (algoritmo de Howard Hinnant).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if month <= 2 { year + 1 } else { year }, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn epoch_stamp_is_1970() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(19_723), (2024, 1, 1));
    }

    #[test]
    fn checkpoint_rollback_restores_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("worlds.json");
        fs::write(&target, b"original").unwrap();
        let cp = save_checkpoint(&target, &dir.path().join("checkpoints"), "test").unwrap();
        atomic_write(&target, b"cambiado").unwrap();
        cp.rollback().unwrap();
        let mut got = Vec::new();
        File::open(&target).unwrap().read_to_end(&mut got).unwrap();
        assert_eq!(got, b"original");
    }

    #[test]
    fn checkpoint_rollback_removes_new_file() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("nuevo.json");
        let cp = save_checkpoint(&target, &dir.path().join("checkpoints"), "test").unwrap();
        atomic_write(&target, b"nuevo").unwrap();
        assert!(target.exists());
        cp.rollback().unwrap();
        assert!(!target.exists());
    }
}
