use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::checkpoint::save_checkpoint;
use crate::error::{Error, Result};
use crate::format::{ALIGN, PREAMBLE_LEN, SUBS_MAGIC, SUBS_VERSION, TRAILER_LEN};
use crate::reader::{subs_path_for, SubsEntry, SubsFile, SubsHeader, SubsSource, UkjFile};

const BUF: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, Serialize)]
pub struct SubsStatus {
    pub ok: bool,
    pub status: String,
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(default)]
    pub entries: usize,
    #[serde(default)]
    pub bytes: u64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repaired: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action: Option<String>,
}

struct Planned {
    tensor: String,
    index: u32,
    ukj_offset: u64,
    offset: u64,
    nbytes: u64,
    sha256: String,
}

fn sha_bytes(value: &str) -> Result<[u8; 32]> {
    let raw = hex::decode(value)
        .map_err(|e| Error::new(format!("checksum hexadecimal inválido: {e}")))?;
    if raw.len() != 32 {
        return Err(Error::new("checksum SHA-256 con longitud distinta de 32"));
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&raw);
    Ok(out)
}

fn make_preamble(
    magic: &[u8; 4],
    version: u32,
    header_len: u64,
    header_sha: &str,
    payload_len: u64,
    payload_sha: &str,
) -> Result<[u8; PREAMBLE_LEN as usize]> {
    let mut out = [0u8; PREAMBLE_LEN as usize];
    out[0..4].copy_from_slice(magic);
    out[4..8].copy_from_slice(&version.to_le_bytes());
    out[8..16].copy_from_slice(&header_len.to_le_bytes());
    out[16..48].copy_from_slice(&sha_bytes(header_sha)?);
    out[48..56].copy_from_slice(&payload_len.to_le_bytes());
    out[56..88].copy_from_slice(&sha_bytes(payload_sha)?);
    Ok(out)
}

fn write_zeros<W: Write>(out: &mut W, mut n: u64) -> Result<()> {
    let zeros = [0u8; 4096];
    while n > 0 {
        let k = std::cmp::min(n, zeros.len() as u64) as usize;
        out.write_all(&zeros[..k])?;
        n -= k as u64;
    }
    Ok(())
}

fn update_zeros(h: &mut Sha256, mut n: u64) {
    let zeros = [0u8; 4096];
    while n > 0 {
        let k = std::cmp::min(n, zeros.len() as u64) as usize;
        h.update(&zeros[..k]);
        n -= k as u64;
    }
}

pub fn extract_subs(
    ukj_path: &Path,
    out_path: Option<&Path>,
    subs_root: Option<&Path>,
    force: bool,
) -> Result<PathBuf> {
    let ukj = UkjFile::open(ukj_path, false, true)?;
    let target = match out_path {
        Some(p) => p.to_path_buf(),
        None => subs_path_for(ukj.model_id(), subs_root)?,
    };
    if target.exists() && !force {
        return Err(Error::new(format!(
            "el .subs ya existe: {} (usa --force)",
            target.display()
        )));
    }
    if let Some(parent) = target.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let mut planned: Vec<Planned> = Vec::new();
    let mut cursor = 0u64;
    for t in &ukj.header.tensors {
        for s in &t.subs {
            cursor = cursor.div_ceil(ALIGN) * ALIGN;
            let ukj_offset = t
                .offset
                .checked_add(s.byte_offset)
                .ok_or_else(|| Error::new("offset de subparámetro desbordado"))?;
            planned.push(Planned {
                tensor: t.name.clone(),
                index: s.index,
                ukj_offset,
                offset: cursor,
                nbytes: s.nbytes,
                sha256: String::new(),
            });
            cursor += s.nbytes;
        }
    }
    let payload_len = cursor;

    // Pasada 1: checksum por subparámetro y del payload .subs.
    let mut src = ukj.container.open_file()?;
    let mut h_payload = Sha256::new();
    let mut pos = 0u64;
    let mut buf = vec![0u8; BUF];
    for e in &mut planned {
        if e.offset > pos {
            update_zeros(&mut h_payload, e.offset - pos);
        }
        src.seek(SeekFrom::Start(ukj.container.payload_offset + e.ukj_offset))?;
        let mut h = Sha256::new();
        let mut remaining = e.nbytes;
        while remaining > 0 {
            let n = std::cmp::min(BUF as u64, remaining) as usize;
            src.read_exact(&mut buf[..n])
                .map_err(|_| Error::new(format!("{}: archivo truncado", ukj.path().display())))?;
            h.update(&buf[..n]);
            h_payload.update(&buf[..n]);
            remaining -= n as u64;
        }
        e.sha256 = hex::encode(h.finalize());
        pos = e.offset + e.nbytes;
    }
    let payload_sha = hex::encode(h_payload.finalize());

    let header = SubsHeader {
        format: "SUBS".to_string(),
        version: SUBS_VERSION,
        created_at: None,
        model_id: ukj.model_id().to_string(),
        source: SubsSource {
            file: ukj
                .path()
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default(),
            size: ukj.container.payload_offset + ukj.container.payload_len + TRAILER_LEN,
            header_sha256: ukj.container.header_sha.clone(),
            payload_sha256: ukj.container.payload_sha.clone(),
            payload_len: ukj.container.payload_len,
        },
        entries: planned
            .iter()
            .map(|e| SubsEntry {
                tensor: e.tensor.clone(),
                index: e.index,
                offset: e.offset,
                nbytes: e.nbytes,
                sha256: e.sha256.clone(),
            })
            .collect(),
    };
    let header_bytes = serde_json::to_vec(&header)?;
    let header_sha = hex::encode(Sha256::digest(&header_bytes));
    let preamble = make_preamble(
        SUBS_MAGIC,
        SUBS_VERSION,
        header_bytes.len() as u64,
        &header_sha,
        payload_len,
        &payload_sha,
    )?;

    let file_name = target
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "subs.subs".to_string());
    let tmp = target.with_file_name(format!(".{file_name}.tmp-{}", std::process::id()));
    let result = (|| -> Result<()> {
        let mut out = File::create(&tmp)?;
        let mut h_file = Sha256::new();
        out.write_all(&preamble)?;
        h_file.update(preamble);
        out.write_all(&header_bytes)?;
        h_file.update(&header_bytes);
        let mut src = ukj.container.open_file()?;
        let mut pos = 0u64;
        for e in &planned {
            if e.offset > pos {
                let pad = e.offset - pos;
                write_zeros(&mut out, pad)?;
                update_zeros(&mut h_file, pad);
            }
            src.seek(SeekFrom::Start(ukj.container.payload_offset + e.ukj_offset))?;
            let mut remaining = e.nbytes;
            while remaining > 0 {
                let n = std::cmp::min(BUF as u64, remaining) as usize;
                src.read_exact(&mut buf[..n])
                    .map_err(|_| Error::new("archivo truncado durante la extracción"))?;
                out.write_all(&buf[..n])?;
                h_file.update(&buf[..n]);
                remaining -= n as u64;
            }
            pos = e.offset + e.nbytes;
        }
        out.write_all(&h_file.finalize())?;
        out.sync_all()?;
        Ok(())
    })();
    if let Err(e) = result {
        let _ = fs::remove_file(&tmp);
        return Err(e);
    }
    let checkpoint_dir = target
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .join("checkpoints");
    let checkpoint = save_checkpoint(&target, &checkpoint_dir, "subs")?;
    fs::rename(&tmp, &target)?;

    let check = verify_subs(ukj_path, Some(&target), subs_root);
    match check {
        Ok(status) if status.ok => {}
        Ok(status) => {
            checkpoint.rollback()?;
            return Err(Error::new(format!(
                "el .subs recién creado no valida: {}; se revirtió al checkpoint",
                status.message
            )));
        }
        Err(e) => {
            checkpoint.rollback()?;
            return Err(Error::new(format!(
                "no se pudo validar el .subs recién creado: {e}; se revirtió al checkpoint"
            )));
        }
    }
    Ok(target)
}

pub fn verify_subs(
    ukj_path: &Path,
    subs_path: Option<&Path>,
    subs_root: Option<&Path>,
) -> Result<SubsStatus> {
    let ukj = UkjFile::open(ukj_path, false, true)?;
    let target = match subs_path {
        Some(p) => p.to_path_buf(),
        None => subs_path_for(ukj.model_id(), subs_root)?,
    };
    let target_str = target.display().to_string();
    if !target.is_file() {
        return Ok(SubsStatus {
            ok: false,
            status: "missing".to_string(),
            path: target_str.clone(),
            model_id: Some(ukj.model_id().to_string()),
            entries: 0,
            bytes: 0,
            message: format!("no existe el .subs: {target_str}"),
            repaired: None,
            action: None,
        });
    }
    let subs = match SubsFile::open(&target, true, true) {
        Ok(s) => s,
        Err(e) => {
            return Ok(SubsStatus {
                ok: false,
                status: "invalid".to_string(),
                path: target_str,
                model_id: Some(ukj.model_id().to_string()),
                entries: 0,
                bytes: 0,
                message: e.to_string(),
                repaired: None,
                action: None,
            })
        }
    };
    if subs.header.model_id != ukj.model_id() {
        return Ok(SubsStatus {
            ok: false,
            status: "mismatch".to_string(),
            path: target_str,
            model_id: Some(ukj.model_id().to_string()),
            entries: 0,
            bytes: 0,
            message: "el .subs pertenece a otro modelo".to_string(),
            repaired: None,
            action: None,
        });
    }
    let src = &subs.header.source;
    if src.header_sha256 != ukj.container.header_sha
        || src.payload_sha256 != ukj.container.payload_sha
        || src.payload_len != ukj.container.payload_len
    {
        return Ok(SubsStatus {
            ok: false,
            status: "desync".to_string(),
            path: target_str,
            model_id: Some(ukj.model_id().to_string()),
            entries: 0,
            bytes: 0,
            message: "el .subs está desincronizado con el .ukj actual".to_string(),
            repaired: None,
            action: None,
        });
    }

    let mut expected: HashMap<(String, u32), (u64, u64)> = HashMap::new();
    for t in &ukj.header.tensors {
        for s in &t.subs {
            let off = t
                .offset
                .checked_add(s.byte_offset)
                .ok_or_else(|| Error::new("offset de subparámetro desbordado"))?;
            expected.insert((t.name.clone(), s.index), (off, s.nbytes));
        }
    }
    let mut seen: HashSet<(String, u32)> = HashSet::new();
    let mut cursor = 0u64;
    let mut f = subs.container.open_file()?;
    for e in &subs.header.entries {
        let key = (e.tensor.clone(), e.index);
        if !seen.insert(key.clone()) {
            return Ok(SubsStatus {
                ok: false,
                status: "invalid".to_string(),
                path: target_str,
                model_id: Some(ukj.model_id().to_string()),
                entries: seen.len(),
                bytes: cursor,
                message: format!("entrada duplicada en .subs: {key:?}"),
                repaired: None,
                action: None,
            });
        }
        let Some((_, nbytes)) = expected.get(&key) else {
            return Ok(SubsStatus {
                ok: false,
                status: "invalid".to_string(),
                path: target_str,
                model_id: Some(ukj.model_id().to_string()),
                entries: seen.len(),
                bytes: cursor,
                message: format!("el .subs tiene un subparámetro que el .ukj no declara: {key:?}"),
                repaired: None,
                action: None,
            });
        };
        if e.nbytes != *nbytes {
            return Ok(SubsStatus {
                ok: false,
                status: "invalid".to_string(),
                path: target_str,
                model_id: Some(ukj.model_id().to_string()),
                entries: seen.len(),
                bytes: cursor,
                message: format!("tamaño distinto para {key:?}"),
                repaired: None,
                action: None,
            });
        }
        if e.offset < cursor || e.offset % ALIGN != 0 {
            return Ok(SubsStatus {
                ok: false,
                status: "invalid".to_string(),
                path: target_str,
                model_id: Some(ukj.model_id().to_string()),
                entries: seen.len(),
                bytes: cursor,
                message: format!("offset inválido en .subs para {key:?}"),
                repaired: None,
                action: None,
            });
        }
        let got = subs.container.hash_range_with(&mut f, e.offset, e.nbytes)?;
        if got != e.sha256 {
            return Ok(SubsStatus {
                ok: false,
                status: "corrupt".to_string(),
                path: target_str,
                model_id: Some(ukj.model_id().to_string()),
                entries: seen.len(),
                bytes: cursor,
                message: format!("bytes corruptos en {} sub {}", key.0, key.1),
                repaired: None,
                action: None,
            });
        }
        cursor = e.offset + e.nbytes;
    }
    if seen.len() != expected.len() {
        return Ok(SubsStatus {
            ok: false,
            status: "incomplete".to_string(),
            path: target_str,
            model_id: Some(ukj.model_id().to_string()),
            entries: seen.len(),
            bytes: cursor,
            message: format!(
                "faltan {} subparámetros en el .subs",
                expected.len() - seen.len()
            ),
            repaired: None,
            action: None,
        });
    }
    Ok(SubsStatus {
        ok: true,
        status: "ok".to_string(),
        path: target_str,
        model_id: Some(ukj.model_id().to_string()),
        entries: seen.len(),
        bytes: cursor,
        message: "el .subs coincide con el .ukj".to_string(),
        repaired: None,
        action: None,
    })
}

pub fn ensure_subs(
    ukj_path: &Path,
    subs_path: Option<&Path>,
    subs_root: Option<&Path>,
) -> Result<SubsStatus> {
    let check = verify_subs(ukj_path, subs_path, subs_root)?;
    if check.ok {
        return Ok(check);
    }
    let existed = Path::new(&check.path).exists();
    let target = extract_subs(ukj_path, Some(Path::new(&check.path)), subs_root, true)?;
    let mut result = verify_subs(ukj_path, Some(&target), subs_root)?;
    result.repaired = Some(true);
    result.action = Some(if existed { "reparado" } else { "creado" }.to_string());
    Ok(result)
}
