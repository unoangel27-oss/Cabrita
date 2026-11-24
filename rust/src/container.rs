use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::error::{Error, Result};
use crate::format::{MAX_HEADER, PREAMBLE_LEN, TRAILER_LEN};

const BUF: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct Container {
    pub path: PathBuf,
    pub header_bytes: Vec<u8>,
    pub header_sha: String,
    pub payload_offset: u64,
    pub payload_len: u64,
    pub payload_sha: String,
    pub file_sha: Option<String>,
}

fn read_u32(b: &[u8]) -> u32 {
    u32::from_le_bytes(b.try_into().expect("4 bytes"))
}

fn read_u64(b: &[u8]) -> u64 {
    u64::from_le_bytes(b.try_into().expect("8 bytes"))
}

pub fn read_container(
    path: &Path,
    magic: &[u8; 4],
    version: u32,
    verify_payload: bool,
    verify_trailer: bool,
) -> Result<Container> {
    let path = path.to_path_buf();
    let mut f = File::open(&path)
        .map_err(|e| Error::new(format!("no se puede abrir {}: {e}", path.display())))?;
    let size = f.metadata()?.len();
    if size < PREAMBLE_LEN + TRAILER_LEN {
        return Err(Error::new(format!(
            "{}: archivo demasiado pequeño para un contenedor",
            path.display()
        )));
    }
    let mut pre = [0u8; PREAMBLE_LEN as usize];
    f.read_exact(&mut pre)
        .map_err(|_| Error::new(format!("{}: preámbulo truncado", path.display())))?;
    if &pre[0..4] != magic {
        return Err(Error::new(format!(
            "{}: no es un contenedor {} (magic {:?})",
            path.display(),
            String::from_utf8_lossy(magic),
            &pre[0..4]
        )));
    }
    let got_version = read_u32(&pre[4..8]);
    if got_version != version {
        return Err(Error::new(format!(
            "{}: versión {got_version} no soportada (se espera {version})",
            path.display()
        )));
    }
    let header_len = read_u64(&pre[8..16]);
    let expected_header_sha = hex::encode(&pre[16..48]);
    let payload_len = read_u64(&pre[48..56]);
    let payload_sha = hex::encode(&pre[56..88]);
    if header_len == 0 || header_len > MAX_HEADER {
        return Err(Error::new(format!(
            "{}: tamaño de cabecera inválido ({header_len})",
            path.display()
        )));
    }
    let payload_offset = PREAMBLE_LEN
        .checked_add(header_len)
        .ok_or_else(|| Error::new("desbordamiento en el offset del payload"))?;
    let expected_size = payload_offset
        .checked_add(payload_len)
        .and_then(|v| v.checked_add(TRAILER_LEN))
        .ok_or_else(|| Error::new("desbordamiento en el tamaño del contenedor"))?;
    if size != expected_size {
        return Err(Error::new(format!(
            "{}: tamaño {size} distinto del declarado {expected_size} \
             (archivo truncado o con bytes extra)",
            path.display()
        )));
    }

    let mut header_bytes = vec![0u8; header_len as usize];
    f.read_exact(&mut header_bytes)
        .map_err(|_| Error::new(format!("{}: cabecera truncada", path.display())))?;
    let header_sha = hex::encode(Sha256::digest(&header_bytes));
    if header_sha != expected_header_sha {
        return Err(Error::new(format!(
            "{}: la cabecera no coincide con su checksum (archivo corrupto o modificado)",
            path.display()
        )));
    }

    let mut file_sha = None;
    if verify_payload || verify_trailer {
        let total = payload_offset + payload_len;
        let mut h_file = Sha256::new();
        let mut h_payload = Sha256::new();
        let mut buf = vec![0u8; BUF];
        f.seek(SeekFrom::Start(0))?;
        let mut pos = 0u64;
        while pos < total {
            let n = std::cmp::min(BUF as u64, total - pos) as usize;
            f.read_exact(&mut buf[..n])
                .map_err(|_| Error::new(format!("{}: archivo truncado", path.display())))?;
            if verify_trailer {
                h_file.update(&buf[..n]);
            }
            if verify_payload && pos + n as u64 > payload_offset {
                let lo = payload_offset.saturating_sub(pos) as usize;
                h_payload.update(&buf[lo..n]);
            }
            pos += n as u64;
        }
        if verify_payload {
            let got = hex::encode(h_payload.finalize());
            if got != payload_sha {
                return Err(Error::new(format!(
                    "{}: el payload no coincide con su checksum (archivo corrupto o modificado)",
                    path.display()
                )));
            }
        }
        if verify_trailer {
            let got = hex::encode(h_file.finalize());
            f.seek(SeekFrom::Start(total))?;
            let mut trailer = [0u8; TRAILER_LEN as usize];
            f.read_exact(&mut trailer)
                .map_err(|_| Error::new(format!("{}: trailer truncado", path.display())))?;
            if hex::encode(trailer) != got {
                return Err(Error::new(format!(
                    "{}: checksum final incorrecto (archivo corrupto o modificado)",
                    path.display()
                )));
            }
            file_sha = Some(got);
        }
    }

    Ok(Container {
        path,
        header_bytes,
        header_sha,
        payload_offset,
        payload_len,
        payload_sha,
        file_sha,
    })
}

impl Container {
    pub fn open_file(&self) -> Result<File> {
        File::open(&self.path)
            .map_err(|e| Error::new(format!("no se puede abrir {}: {e}", self.path.display())))
    }

    pub fn hash_range_with(&self, f: &mut File, offset: u64, len: u64) -> Result<String> {
        f.seek(SeekFrom::Start(self.payload_offset + offset))?;
        let mut h = Sha256::new();
        let mut remaining = len;
        let mut buf = vec![0u8; BUF];
        while remaining > 0 {
            let n = std::cmp::min(BUF as u64, remaining) as usize;
            f.read_exact(&mut buf[..n])
                .map_err(|_| Error::new(format!("{}: archivo truncado", self.path.display())))?;
            h.update(&buf[..n]);
            remaining -= n as u64;
        }
        Ok(hex::encode(h.finalize()))
    }

    pub fn copy_range_with<W: Write>(
        &self,
        f: &mut File,
        offset: u64,
        len: u64,
        out: &mut W,
    ) -> Result<()> {
        f.seek(SeekFrom::Start(self.payload_offset + offset))?;
        let mut remaining = len;
        let mut buf = vec![0u8; BUF];
        while remaining > 0 {
            let n = std::cmp::min(BUF as u64, remaining) as usize;
            f.read_exact(&mut buf[..n])
                .map_err(|_| Error::new(format!("{}: archivo truncado", self.path.display())))?;
            out.write_all(&buf[..n])?;
            remaining -= n as u64;
        }
        Ok(())
    }

    pub fn hash_range(&self, offset: u64, len: u64) -> Result<String> {
        let mut f = self.open_file()?;
        self.hash_range_with(&mut f, offset, len)
    }
}
