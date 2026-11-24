use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::container::{read_container, Container};
use crate::error::{Error, Result};
use crate::format::{validate_header, Header, ALIGN, MAGIC, SUBS_MAGIC, SUBS_VERSION, VERSION};

#[derive(Debug, Clone)]
pub struct UkjFile {
    pub container: Container,
    pub header: Header,
}

impl UkjFile {
    pub fn open(path: &Path, verify_payload: bool, verify_trailer: bool) -> Result<Self> {
        let container = read_container(path, MAGIC, VERSION, verify_payload, verify_trailer)?;
        let header: Header = serde_json::from_slice(&container.header_bytes)
            .map_err(|e| Error::new(format!("{}: cabecera JSON inválida: {e}", path.display())))?;
        validate_header(&header, container.payload_len)?;
        Ok(Self { container, header })
    }

    pub fn path(&self) -> &Path {
        &self.container.path
    }

    pub fn payload_len(&self) -> u64 {
        self.container.payload_len
    }

    pub fn model_id(&self) -> &str {
        &self.header.model.id
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubsSource {
    #[serde(default)]
    pub file: String,
    #[serde(default)]
    pub size: u64,
    pub header_sha256: String,
    pub payload_sha256: String,
    pub payload_len: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubsEntry {
    pub tensor: String,
    pub index: u32,
    pub offset: u64,
    pub nbytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubsHeader {
    pub format: String,
    pub version: u32,
    #[serde(default)]
    pub created_at: Option<String>,
    pub model_id: String,
    pub source: SubsSource,
    pub entries: Vec<SubsEntry>,
}

#[derive(Debug, Clone)]
pub struct SubsFile {
    pub container: Container,
    pub header: SubsHeader,
}

impl SubsFile {
    pub fn open(path: &Path, verify_payload: bool, verify_trailer: bool) -> Result<Self> {
        let container = read_container(
            path,
            SUBS_MAGIC,
            SUBS_VERSION,
            verify_payload,
            verify_trailer,
        )?;
        let header: SubsHeader = serde_json::from_slice(&container.header_bytes)
            .map_err(|e| Error::new(format!("{}: cabecera .subs inválida: {e}", path.display())))?;
        if header.format != "SUBS" {
            return Err(Error::new(format!(
                "{}: la cabecera no declara format=SUBS",
                path.display()
            )));
        }
        if header.version != SUBS_VERSION {
            return Err(Error::new(format!(
                "{}: versión .subs {} no soportada",
                path.display(),
                header.version
            )));
        }
        if header.model_id.is_empty() {
            return Err(Error::new(format!(
                "{}: .subs sin model_id",
                path.display()
            )));
        }
        let mut cursor = 0u64;
        for e in &header.entries {
            if e.nbytes == 0 || e.offset % ALIGN != 0 || e.offset < cursor {
                return Err(Error::new(format!(
                    "{}: entrada .subs inválida para {} sub {}",
                    path.display(),
                    e.tensor,
                    e.index
                )));
            }
            if e.sha256.len() != 64 {
                return Err(Error::new(format!(
                    "{}: sha256 inválido en {} sub {}",
                    path.display(),
                    e.tensor,
                    e.index
                )));
            }
            cursor = e
                .offset
                .checked_add(e.nbytes)
                .ok_or_else(|| Error::new("desbordamiento en .subs"))?;
            if cursor > container.payload_len {
                return Err(Error::new(format!(
                    "{}: entrada .subs fuera del payload",
                    path.display()
                )));
            }
        }
        Ok(Self { container, header })
    }

    pub fn path(&self) -> &Path {
        &self.container.path
    }
}

pub fn default_subs_root() -> PathBuf {
    if let Some(root) = std::env::var_os("CABRITA_HOME") {
        return PathBuf::from(root).join("models");
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("USERPROFILE").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".cabrita").join("models")
}

pub fn subs_path_for(model_id: &str, root: Option<&Path>) -> Result<PathBuf> {
    if model_id.is_empty()
        || model_id.contains('/')
        || model_id.contains('\\')
        || model_id == "."
        || model_id == ".."
    {
        return Err(Error::new(format!(
            "model_id inválido para la ruta .subs: {model_id:?}"
        )));
    }
    let root = root
        .map(Path::to_path_buf)
        .unwrap_or_else(default_subs_root);
    Ok(root.join(model_id).join("subs.subs"))
}
