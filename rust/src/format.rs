use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

pub const MAGIC: &[u8; 4] = b"UKJ1";
pub const VERSION: u32 = 1;
pub const PREAMBLE_LEN: u64 = 88;
pub const TRAILER_LEN: u64 = 32;
pub const ALIGN: u64 = 64;
pub const SUBS_MAGIC: &[u8; 4] = b"SUBS";
pub const SUBS_VERSION: u32 = 1;
pub const MAX_HEADER: u64 = 256 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Header {
    pub format: String,
    pub version: u32,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub generator: Option<String>,
    #[serde(default)]
    pub layout: serde_json::Value,
    #[serde(default)]
    pub model: Model,
    pub payload: Payload,
    pub routes: Vec<Route>,
    pub operators: Vec<serde_json::Value>,
    pub densities: Densities,
    pub tensors: Vec<Tensor>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Model {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub architecture: Option<String>,
    #[serde(default)]
    pub quantization: Option<String>,
    #[serde(default)]
    pub tensor_count: Option<u64>,
    #[serde(default)]
    pub parameter_count: Option<u64>,
    #[serde(default)]
    pub metadata_sha256: Option<String>,
    #[serde(default)]
    pub source: serde_json::Value,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Payload {
    pub length: u64,
    #[serde(default)]
    pub alignment: u64,
    #[serde(default)]
    pub sha256: Option<String>,
    /// Tensor original cuando esta entrada nació de `continue parameter`.
    #[serde(default)]
    pub continued_from: Option<String>,
    /// Parámetro independiente sin subparámetros propios.
    #[serde(default)]
    pub independent: bool,
    /// Offset del corte dentro del tensor original.
    #[serde(default)]
    pub source_offset: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Route {
    pub id: String,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub auto: bool,
    #[serde(default)]
    pub tensors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Densities {
    pub default: String,
    #[serde(default)]
    pub allowed: Vec<String>,
    #[serde(default)]
    pub aliases: BTreeMap<String, String>,
    #[serde(default)]
    pub declarations: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    pub name: String,
    #[serde(default)]
    pub dims: Vec<u64>,
    #[serde(default)]
    pub ggml_type: u32,
    #[serde(rename = "type", default)]
    pub type_name: String,
    pub offset: u64,
    pub nbytes: u64,
    #[serde(default)]
    pub rows: Option<u64>,
    #[serde(default)]
    pub route: Option<String>,
    #[serde(default)]
    pub density: Option<String>,
    pub subs: Vec<Sub>,
    #[serde(default)]
    pub sha256: Option<String>,
    /// Tensor original cuando esta entrada nació de `continue parameter`.
    #[serde(default)]
    pub continued_from: Option<String>,
    /// Parámetro independiente sin subparámetros propios.
    #[serde(default)]
    pub independent: bool,
    /// Offset del corte dentro del tensor original.
    #[serde(default)]
    pub source_offset: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sub {
    pub index: u32,
    #[serde(default)]
    pub row_start: u64,
    #[serde(default)]
    pub row_count: u64,
    pub byte_offset: u64,
    pub nbytes: u64,
}

pub fn parse_density(code: &str) -> Result<u32> {
    let lower = code.trim().to_ascii_lowercase();
    let canonical = match lower.as_str() {
        "p_p" | "p" => "p1_p".to_string(),
        _ => lower,
    };
    let rest = canonical
        .strip_prefix('p')
        .and_then(|s| s.strip_suffix("_p"))
        .ok_or_else(|| {
            Error::new(format!(
                "densidad '{code}' inválida; usa p1_p…p8_p (p_p equivale a p1_p)"
            ))
        })?;
    let n: u32 = rest.parse().map_err(|_| {
        Error::new(format!(
            "densidad '{code}' inválida; usa p1_p…p8_p (p_p equivale a p1_p)"
        ))
    })?;
    if !(1..=12).contains(&n) || n.to_string() != rest {
        return Err(Error::new(format!(
            "densidad '{code}' inválida; usa p1_p…p8_p; p9_p…p12_p son solo para el experimento >p8_p"
        )));
    }
    Ok(n)
}

pub fn validate_header(header: &Header, payload_len: u64) -> Result<()> {
    if header.format != "UKJ" {
        return Err(Error::new("la cabecera no declara format=UKJ"));
    }
    if header.version != VERSION {
        return Err(Error::new(format!(
            "versión de esquema {} no soportada (se espera {VERSION})",
            header.version
        )));
    }
    if header.tensors.is_empty() {
        return Err(Error::new("la cabecera no contiene tensores"));
    }
    if header.payload.length != payload_len {
        return Err(Error::new(format!(
            "payload.length {} no coincide con el contenedor {payload_len}",
            header.payload.length
        )));
    }

    let mut names: HashSet<&str> = HashSet::new();
    let mut intervals: Vec<(u64, u64, &str)> = Vec::new();
    for t in &header.tensors {
        if t.name.is_empty() || !names.insert(t.name.as_str()) {
            return Err(Error::new(format!(
                "tensor inválido o duplicado: {:?}",
                t.name
            )));
        }
        if t.nbytes == 0 {
            return Err(Error::new(format!("tensor '{}': nbytes inválido", t.name)));
        }
        let end = t
            .offset
            .checked_add(t.nbytes)
            .ok_or_else(|| Error::new(format!("tensor '{}': offset desbordado", t.name)))?;
        if end > payload_len {
            return Err(Error::new(format!(
                "tensor '{}' fuera del payload ({end} > {payload_len})",
                t.name
            )));
        }
        intervals.push((t.offset, end, t.name.as_str()));

        if t.subs.is_empty() {
            return Err(Error::new(format!(
                "tensor '{}': sin subparámetros",
                t.name
            )));
        }
        let mut cursor = 0u64;
        for (i, s) in t.subs.iter().enumerate() {
            if s.index as usize != i {
                return Err(Error::new(format!(
                    "tensor '{}': índices de subparámetro no consecutivos",
                    t.name
                )));
            }
            if s.byte_offset != cursor {
                return Err(Error::new(format!(
                    "tensor '{}': subparámetros no contiguos",
                    t.name
                )));
            }
            if s.nbytes == 0 {
                return Err(Error::new(format!(
                    "tensor '{}': subparámetro con tamaño inválido",
                    t.name
                )));
            }
            cursor += s.nbytes;
        }
        if cursor != t.nbytes {
            return Err(Error::new(format!(
                "tensor '{}': los subparámetros suman {cursor} y el tensor {}",
                t.name, t.nbytes
            )));
        }
        if let Some(density) = &t.density {
            let p = parse_density(density)?;
            if p as usize != t.subs.len() {
                return Err(Error::new(format!(
                    "tensor '{}': densidad {density} no coincide con {} subparámetros",
                    t.name,
                    t.subs.len()
                )));
            }
        }
    }

    intervals.sort_by_key(|x| x.0);
    let mut prev_end = 0u64;
    let mut prev_name = "";
    for (start, end, name) in intervals {
        if start < prev_end {
            return Err(Error::new(format!(
                "tensores solapados: '{prev_name}' y '{name}'"
            )));
        }
        prev_end = end;
        prev_name = name;
    }

    if header.routes.is_empty() {
        return Err(Error::new("la cabecera no contiene rutas"));
    }
    let mut route_ids: HashSet<&str> = HashSet::new();
    let mut referenced: HashSet<&str> = HashSet::new();
    for r in &header.routes {
        if r.id.is_empty() || !route_ids.insert(r.id.as_str()) {
            return Err(Error::new(format!("ruta inválida o duplicada: {:?}", r.id)));
        }
        for name in &r.tensors {
            if !names.contains(name.as_str()) {
                return Err(Error::new(format!(
                    "ruta '{}' referencia un tensor inexistente: '{name}'",
                    r.id
                )));
            }
            if !referenced.insert(name.as_str()) {
                return Err(Error::new(format!(
                    "tensor '{name}' asignado a más de una ruta"
                )));
            }
        }
    }
    if referenced.len() != names.len() {
        let missing = names.len() - referenced.len();
        return Err(Error::new(format!(
            "{missing} tensores sin ruta en la cabecera"
        )));
    }

    if header.operators.is_empty() {
        return Err(Error::new("la cabecera no contiene operadores"));
    }
    for op in &header.operators {
        if op.get("name").and_then(|v| v.as_str()).is_none() {
            return Err(Error::new("operador sin nombre"));
        }
    }
    parse_density(&header.densities.default)?;
    Ok(())
}
