//! Respaldos, restauración con rollback y reentrenamiento ligero de mundos.
//!
//! La taxonomía de mundos es un JSON pequeño. Estas operaciones nunca dejan el
//! archivo a medias: escriben un temporal, validan y, si algo falla, revierten
//! al checkpoint anterior.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::checkpoint::{atomic_write, save_checkpoint, sha256_bytes, utc_stamp};
use crate::error::{Error, Result};
use crate::router::{evaluate_holdout, ActivationPolicy, TinyRouter, WorldTaxonomy};

#[derive(Debug, Clone, Serialize)]
pub struct WorldSummary {
    pub worlds: usize,
    pub examples: usize,
    pub feature_dim: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct BackupInfo {
    pub source: String,
    pub backup: String,
    pub manifest: String,
    pub sha256: String,
    pub bytes: u64,
    pub created_at: String,
    pub worlds: usize,
    pub examples: usize,
    pub feature_dim: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct RestoreInfo {
    pub target: String,
    pub backup: String,
    pub checkpoint: Option<String>,
    pub sha256: String,
    pub bytes: u64,
    pub worlds: usize,
    pub examples: usize,
    pub feature_dim: usize,
    pub rolled_back: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct RetrainInfo {
    pub source: String,
    pub output: String,
    pub checkpoint: Option<String>,
    pub sha256: String,
    pub bytes: u64,
    pub worlds: usize,
    pub examples: usize,
    pub feature_dim: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accuracy: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub train_examples: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub test_examples: Option<usize>,
}

fn summarize(tax: &WorldTaxonomy) -> WorldSummary {
    WorldSummary {
        worlds: tax.worlds.len(),
        examples: tax.worlds.iter().map(|w| w.examples.len()).sum(),
        feature_dim: tax.feature_dim,
    }
}

fn default_dir(path: &Path, name: &str) -> PathBuf {
    path.parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .join(name)
}

fn file_stem(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "worlds".to_string())
}

fn manifest_path(backup: &Path) -> PathBuf {
    let name = backup
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "backup.json".to_string());
    backup.with_file_name(format!("{name}.manifest.json"))
}

fn unique_path(dir: &Path, stem: &str, stamp: &str) -> PathBuf {
    let mut candidate = dir.join(format!("{stem}-{stamp}.json"));
    let mut n = 1usize;
    while candidate.exists() {
        candidate = dir.join(format!("{stem}-{stamp}-{n}.json"));
        n += 1;
    }
    candidate
}

/// Copia la taxonomía a un backup con manifiesto SHA-256.
pub fn backup_worlds(source: &Path, out_dir: Option<&Path>) -> Result<BackupInfo> {
    let bytes = fs::read(source)
        .map_err(|e| Error::new(format!("no se puede leer {}: {e}", source.display())))?;
    let tax = WorldTaxonomy::load(source)?;
    let summary = summarize(&tax);
    let dir = out_dir
        .map(Path::to_path_buf)
        .unwrap_or_else(|| default_dir(source, "backups"));
    fs::create_dir_all(&dir).map_err(|e| {
        Error::new(format!(
            "no se puede crear el directorio de backups {}: {e}",
            dir.display()
        ))
    })?;
    let stamp = utc_stamp();
    let backup = unique_path(&dir, &file_stem(source), &stamp);
    let sha256 = sha256_bytes(&bytes);

    let mut info = BackupInfo {
        source: source.display().to_string(),
        backup: backup.display().to_string(),
        manifest: manifest_path(&backup).display().to_string(),
        sha256,
        bytes: bytes.len() as u64,
        created_at: stamp,
        worlds: summary.worlds,
        examples: summary.examples,
        feature_dim: summary.feature_dim,
    };
    let manifest = manifest_path(&backup);
    let manifest_bytes = serde_json::to_vec_pretty(&info)?;
    atomic_write(&backup, &bytes)?;
    if let Err(e) = atomic_write(&manifest, &manifest_bytes) {
        let _ = fs::remove_file(&backup);
        return Err(e);
    }
    info.manifest = manifest.display().to_string();
    Ok(info)
}

fn verify_manifest(backup: &Path, bytes: &[u8]) -> Result<()> {
    let manifest = manifest_path(backup);
    if !manifest.is_file() {
        // Un backup sin manifiesto no es verificable: restaurarlo saltaría la
        // comprobación de integridad (misma familia que "ausente = válido").
        return Err(Error::new(format!(
            "el backup {} no tiene manifiesto {}; no se puede verificar su integridad",
            backup.display(),
            manifest.display()
        )));
    }
    let raw = fs::read(&manifest).map_err(|e| {
        Error::new(format!(
            "no se puede leer el manifiesto {}: {e}",
            manifest.display()
        ))
    })?;
    let value: serde_json::Value = serde_json::from_slice(&raw)
        .map_err(|e| Error::new(format!("manifiesto inválido {}: {e}", manifest.display())))?;
    let expected = value
        .get("sha256")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::new("el manifiesto no declara sha256"))?;
    let got = sha256_bytes(bytes);
    if got != expected {
        return Err(Error::new(format!(
            "el backup {} no coincide con su manifiesto (sha256 {got} != {expected})",
            backup.display()
        )));
    }
    Ok(())
}

/// Restaura un backup validando antes y después; si falla, revierte.
pub fn restore_worlds(
    target: &Path,
    backup: &Path,
    checkpoint_dir: Option<&Path>,
) -> Result<RestoreInfo> {
    let bytes = fs::read(backup)
        .map_err(|e| Error::new(format!("no se puede leer {}: {e}", backup.display())))?;
    verify_manifest(backup, &bytes)?;
    let tax = WorldTaxonomy::load(backup).map_err(|e| {
        Error::new(format!(
            "el backup {} no es una taxonomía válida: {e}",
            backup.display()
        ))
    })?;
    let summary = summarize(&tax);
    let dir = checkpoint_dir
        .map(Path::to_path_buf)
        .unwrap_or_else(|| default_dir(target, "checkpoints"));
    let cp = save_checkpoint(target, &dir, "restore")?;
    let sha256 = sha256_bytes(&bytes);

    let result = (|| -> Result<()> {
        atomic_write(target, &bytes)?;
        WorldTaxonomy::load(target).map_err(|e| {
            Error::new(format!(
                "la taxonomía restaurada {} no valida: {e}",
                target.display()
            ))
        })?;
        Ok(())
    })();
    match result {
        Ok(()) => Ok(RestoreInfo {
            target: target.display().to_string(),
            backup: backup.display().to_string(),
            checkpoint: cp.backup().map(|p| p.display().to_string()),
            sha256,
            bytes: bytes.len() as u64,
            worlds: summary.worlds,
            examples: summary.examples,
            feature_dim: summary.feature_dim,
            rolled_back: false,
        }),
        Err(e) => {
            let rollback = cp.rollback();
            let checkpoint = cp.backup().map(|p| p.display().to_string());
            match rollback {
                Ok(()) => Err(Error::new(format!(
                    "restauración falló: {e}; se revirtió al checkpoint {}",
                    checkpoint.as_deref().unwrap_or("(archivo nuevo eliminado)")
                ))),
                Err(rb) => Err(Error::new(format!(
                    "restauración falló: {e}; además falló el rollback: {rb}"
                ))),
            }
        }
    }
}

/// Reentrena el router lineal y escribe el artefacto de forma atómica.
pub fn retrain_worlds(
    source: &Path,
    out: Option<&Path>,
    eval: bool,
    holdout_every: usize,
) -> Result<RetrainInfo> {
    let tax = WorldTaxonomy::load(source)?;
    let summary = summarize(&tax);
    let router = TinyRouter::train(&tax)?;
    let bytes = serde_json::to_vec_pretty(&router)?;
    let output = out
        .map(Path::to_path_buf)
        .unwrap_or_else(|| source.with_extension("router.json"));
    let dir = default_dir(&output, "checkpoints");
    let cp = save_checkpoint(&output, &dir, "retrain")?;
    let result = (|| -> Result<()> {
        atomic_write(&output, &bytes)?;
        let raw = fs::read(&output)?;
        serde_json::from_slice::<TinyRouter>(&raw).map_err(|e| {
            Error::new(format!(
                "el router reentrenado {} no se puede releer: {e}",
                output.display()
            ))
        })?;
        Ok(())
    })();
    if let Err(e) = result {
        cp.rollback()?;
        return Err(Error::new(format!(
            "reentrenamiento falló: {e}; se revirtió el artefacto anterior"
        )));
    }

    let (accuracy, train_examples, test_examples) = if eval {
        let report = evaluate_holdout(&tax, holdout_every, &ActivationPolicy::default())?;
        (
            Some(report.accuracy),
            Some(report.train_examples),
            Some(report.test_examples),
        )
    } else {
        (None, None, None)
    };
    Ok(RetrainInfo {
        source: source.display().to_string(),
        output: output.display().to_string(),
        checkpoint: cp.backup().map(|p| p.display().to_string()),
        sha256: sha256_bytes(&bytes),
        bytes: bytes.len() as u64,
        worlds: summary.worlds,
        examples: summary.examples,
        feature_dim: summary.feature_dim,
        accuracy,
        train_examples,
        test_examples,
    })
}
