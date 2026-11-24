//! Operadores de Fase 0: `stop-subparameter`, `continue parameter` y tics.
//!
//! Los tics son activaciones de subparámetros (expertos) contadas en una
//! ventana deslizante de un segundo. `stop-subparameter` no cambia la salida:
//! decide qué páginas se **despliegan** (prefetch) y cuáles se sueltan
//! (offload) según el nivel de tics. Los pesos siguen en `.ukj`/`.subs` y se
//! leen por mmap cuando hacen falta, así que un stop nunca deja el modelo
//! inservible.

use std::collections::VecDeque;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::error::{Error, Result};
use crate::model::TensorInfo;

/// Configuración de los operadores. Los valores por defecto son los de la spec.
#[derive(Debug, Clone, Serialize)]
pub struct OperatorConfig {
    pub stop_subparameter: bool,
    pub continue_parameter: bool,
    /// Nivel normal de tics (activaciones/s). Por debajo se sueltan páginas.
    pub tics_normal: f32,
    /// Nivel anómalo de tics. Al alcanzarlo se despliegan los subparámetros.
    pub tics_anomalous: f32,
    /// Ventana de medida en milisegundos.
    pub window_ms: u64,
}

impl Default for OperatorConfig {
    fn default() -> Self {
        Self {
            stop_subparameter: true,
            continue_parameter: true,
            tics_normal: 2.0,
            tics_anomalous: 20.0,
            window_ms: 1000,
        }
    }
}

impl OperatorConfig {
    pub fn validate(&self) -> Result<()> {
        if self.tics_normal < 0.0 || self.tics_anomalous <= self.tics_normal {
            return Err(Error::new(format!(
                "umbrales de tics inválidos: normal={} anómalo={} (debe ser anómalo > normal)",
                self.tics_normal, self.tics_anomalous
            )));
        }
        if self.window_ms == 0 {
            return Err(Error::new("la ventana de tics no puede ser 0"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DeploymentState {
    /// Sin despliegue: no se hace prefetch y las páginas se sueltan.
    Stopped,
    /// Zona intermedia: se mantiene el estado anterior (histéresis).
    Holding,
    /// Desplegado: se hace prefetch de los subparámetros de la capa.
    Deployed,
}

impl DeploymentState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stopped => "stopped",
            Self::Holding => "holding",
            Self::Deployed => "deployed",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TicsTransition {
    pub ts_ms: u64,
    pub layer: usize,
    pub from: DeploymentState,
    pub to: DeploymentState,
    pub tics_per_s: f32,
    pub action: &'static str,
}

/// Contador de tics por capa con ventana deslizante e histéresis.
pub struct TicsTracker {
    cfg: OperatorConfig,
    layers: usize,
    window: Duration,
    events: Vec<VecDeque<Instant>>,
    state: Vec<DeploymentState>,
    transitions: Vec<TicsTransition>,
}

impl TicsTracker {
    pub fn new(cfg: OperatorConfig, layers: usize) -> Result<Self> {
        cfg.validate()?;
        if layers == 0 {
            return Err(Error::new("el tracker de tics necesita al menos una capa"));
        }
        Ok(Self {
            window: Duration::from_millis(cfg.window_ms),
            cfg,
            layers,
            events: (0..layers).map(|_| VecDeque::new()).collect(),
            state: vec![DeploymentState::Stopped; layers],
            transitions: Vec::new(),
        })
    }

    pub fn config(&self) -> &OperatorConfig {
        &self.cfg
    }

    pub fn layers(&self) -> usize {
        self.layers
    }

    pub fn state(&self, layer: usize) -> DeploymentState {
        self.state[layer]
    }

    pub fn transitions(&self) -> &[TicsTransition] {
        &self.transitions
    }

    pub fn into_transitions(self) -> Vec<TicsTransition> {
        self.transitions
    }

    fn prune(&mut self, layer: usize, now: Instant) {
        let events = &mut self.events[layer];
        while let Some(front) = events.front() {
            if now.duration_since(*front) > self.window {
                events.pop_front();
            } else {
                break;
            }
        }
    }

    pub fn rate(&mut self, layer: usize, now: Instant) -> f32 {
        self.prune(layer, now);
        let secs = self.window.as_secs_f64().max(1e-6);
        self.events[layer].len() as f32 / secs as f32
    }

    /// Registra `count` activaciones de subparámetros de una capa.
    pub fn record(&mut self, layer: usize, count: usize, now: Instant) -> Option<TicsTransition> {
        if layer >= self.layers {
            return None;
        }
        for _ in 0..count {
            self.events[layer].push_back(now);
        }
        self.evaluate(layer, now)
    }

    /// Reevalúa una capa sin registrar activaciones (para detectar el idle).
    pub fn tick(&mut self, layer: usize, now: Instant) -> Option<TicsTransition> {
        if layer >= self.layers {
            return None;
        }
        self.evaluate(layer, now)
    }

    fn evaluate(&mut self, layer: usize, now: Instant) -> Option<TicsTransition> {
        let rate = self.rate(layer, now);
        let from = self.state[layer];
        let to = if rate >= self.cfg.tics_anomalous {
            DeploymentState::Deployed
        } else if rate <= self.cfg.tics_normal {
            DeploymentState::Stopped
        } else {
            DeploymentState::Holding
        };
        // La zona intermedia conserva el estado anterior; solo se registra un
        // cambio real de despliegue.
        if to == DeploymentState::Holding {
            return None;
        }
        if to == from {
            return None;
        }
        self.state[layer] = to;
        let action = match to {
            DeploymentState::Deployed => "deploy",
            DeploymentState::Stopped => "offload",
            DeploymentState::Holding => "hold",
        };
        let transition = TicsTransition {
            ts_ms: unix_ms(),
            layer,
            from,
            to,
            tics_per_s: rate,
            action,
        };
        self.transitions.push(transition.clone());
        Some(transition)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct OperatorEvent {
    pub ts_ms: u64,
    pub operator: String,
    pub target: String,
    pub tics_per_s: f32,
    pub state: String,
    pub action: String,
    pub rss_bytes: u64,
}

/// Log de sesión en JSONL. Cada línea es un evento de operador.
pub struct SessionLog {
    path: PathBuf,
    file: std::cell::RefCell<File>,
}

impl SessionLog {
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(dir) = path.parent() {
            if !dir.as_os_str().is_empty() {
                fs::create_dir_all(dir)
                    .map_err(|e| Error::new(format!("no se puede crear {}: {e}", dir.display())))?;
            }
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| Error::new(format!("no se puede abrir {}: {e}", path.display())))?;
        Ok(Self {
            path,
            file: std::cell::RefCell::new(file),
        })
    }

    /// Log de operadores junto al modelo: `<modelo>.ops.jsonl`.
    pub fn path_for_model(model: impl AsRef<Path>) -> PathBuf {
        let model = model.as_ref();
        let name = model
            .file_name()
            .map(|n| format!("{}.ops.jsonl", n.to_string_lossy()))
            .unwrap_or_else(|| "cabrita.ops.jsonl".to_string());
        model.with_file_name(name)
    }

    pub fn default_path() -> PathBuf {
        let dir = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".cabrita")
            .join("logs");
        dir.join(format!("session-{}.jsonl", unix_ms()))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn write(&self, event: &OperatorEvent) -> Result<()> {
        let line = serde_json::to_string(event)
            .map_err(|e| Error::new(format!("no se puede serializar el evento: {e}")))?;
        let mut file = self.file.borrow_mut();
        writeln!(file, "{line}")
            .and_then(|_| file.flush())
            .map_err(|e| Error::new(format!("no se puede escribir {}: {e}", self.path.display())))
    }
}

/// Lee un log JSONL y devuelve sus eventos.
pub fn read_log(path: impl AsRef<Path>) -> Result<Vec<serde_json::Value>> {
    let text = fs::read_to_string(path.as_ref())
        .map_err(|e| Error::new(format!("no se puede leer {}: {e}", path.as_ref().display())))?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line).map_err(|e| {
            Error::new(format!(
                "{}: línea {} no es JSON válido: {e}",
                path.as_ref().display(),
                i + 1
            ))
        })?;
        out.push(value);
    }
    Ok(out)
}

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Una rebanada independiente de un tensor irregular (definición 46).
#[derive(Debug, Clone, Serialize)]
pub struct ContinuedSlice {
    pub index: usize,
    pub name: String,
    /// Offset relativo al tensor original.
    pub byte_offset: u64,
    pub nbytes: u64,
    /// Filas de la rebanada si el corte es por filas.
    pub rows: Option<usize>,
    /// Cada rebanada es un parámetro independiente sin subparámetros propios.
    pub independent: bool,
}

/// `continue parameter` (definición 46): reparte un tensor irregular en
/// parámetros independientes sin subparámetros. Respeta filas completas y
/// bloques GGUF; la concatenación reproduce el tensor original.
pub fn continue_parameter_slices(info: &TensorInfo, parts: usize) -> Result<Vec<ContinuedSlice>> {
    if parts < 1 {
        return Err(Error::new("continue parameter necesita al menos 1 parte"));
    }
    if info.nbytes == 0 {
        return Err(Error::new(format!(
            "tensor '{}': no se puede repartir un tensor vacío",
            info.name
        )));
    }

    let row_bytes = info.row_bytes().ok();
    if let (Some(rb), true) = (row_bytes, info.rows >= parts && info.rows > 0) {
        if rb > 0 && (info.nbytes as usize) == rb * info.rows {
            let base = info.rows / parts;
            let rem = info.rows % parts;
            let mut out = Vec::with_capacity(parts);
            let mut row = 0usize;
            for k in 0..parts {
                let nrows = base + usize::from(k < rem);
                if nrows == 0 {
                    continue;
                }
                out.push(ContinuedSlice {
                    index: out.len(),
                    name: format!("{}.c{}", info.name, out.len()),
                    byte_offset: (row * rb) as u64,
                    nbytes: (nrows * rb) as u64,
                    rows: Some(nrows),
                    independent: true,
                });
                row += nrows;
            }
            if !out.is_empty() {
                return Ok(out);
            }
        }
    }

    // Corte por bloques GGUF completos (no rompe la cuantización).
    let block = info.qt.block_bytes().max(1);
    let blocks = info.nbytes as usize / block;
    if blocks == 0 {
        return Err(Error::new(format!(
            "tensor '{}': {} bytes no contienen un bloque completo de {} bytes",
            info.name, info.nbytes, block
        )));
    }
    let effective = parts.min(blocks);
    let base = blocks / effective;
    let rem = blocks % effective;
    let mut out = Vec::with_capacity(effective);
    let mut offset_blocks = 0usize;
    for k in 0..effective {
        let nblocks = base + usize::from(k < rem);
        if nblocks == 0 {
            continue;
        }
        out.push(ContinuedSlice {
            index: out.len(),
            name: format!("{}.c{}", info.name, out.len()),
            byte_offset: (offset_blocks * block) as u64,
            nbytes: (nblocks * block) as u64,
            rows: None,
            independent: true,
        });
        offset_blocks += nblocks;
    }
    // Resto que no completa un bloque (solo posible en F32/F16 con tamaños
    // irregulares): se añade a la última rebanada.
    let remainder = info.nbytes - (blocks * block) as u64;
    if remainder > 0 {
        if let Some(last) = out.last_mut() {
            last.nbytes += remainder;
        }
    }
    let total: u64 = out.iter().map(|s| s.nbytes).sum();
    if total != info.nbytes {
        return Err(Error::new(format!(
            "continue parameter no reconstruye '{}': {total} != {}",
            info.name, info.nbytes
        )));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::QuantType;

    fn cfg(normal: f32, anomalous: f32) -> OperatorConfig {
        OperatorConfig {
            stop_subparameter: true,
            continue_parameter: true,
            tics_normal: normal,
            tics_anomalous: anomalous,
            window_ms: 1000,
        }
    }

    #[test]
    fn tics_2_and_20_thresholds() {
        let mut t = TicsTracker::new(cfg(2.0, 20.0), 1).unwrap();
        let start = Instant::now();

        // 1 activación/s: normal, no despliega.
        for s in 0..3 {
            let now = start + Duration::from_secs(s);
            assert!(t.record(0, 1, now).is_none());
            assert_eq!(t.state(0), DeploymentState::Stopped);
        }

        // 5 activaciones/s: zona intermedia, mantiene el estado.
        for s in 3..5 {
            let now = start + Duration::from_secs(s);
            assert!(t.record(0, 5, now).is_none());
            assert_eq!(t.state(0), DeploymentState::Stopped);
        }

        // 25 activaciones/s: supera 20/s, despliega.
        let tr = t
            .record(0, 25, start + Duration::from_secs(5))
            .expect("debe desplegar");
        assert_eq!(tr.to, DeploymentState::Deployed);
        assert_eq!(tr.action, "deploy");
        assert!(tr.tics_per_s >= 20.0);

        // Tras 2 s sin actividad: suelta.
        let tr = t
            .tick(0, start + Duration::from_secs(8))
            .expect("debe soltar");
        assert_eq!(tr.to, DeploymentState::Stopped);
        assert_eq!(tr.action, "offload");
        assert_eq!(t.transitions().len(), 2);
    }

    #[test]
    fn tics_validate_thresholds() {
        assert!(TicsTracker::new(cfg(20.0, 2.0), 1).is_err());
        assert!(TicsTracker::new(cfg(2.0, 2.0), 1).is_err());
    }

    fn tensor(name: &str, qt: QuantType, rows: usize, cols: usize) -> TensorInfo {
        let nbytes = qt.row_bytes(cols).unwrap() as u64 * rows as u64;
        TensorInfo {
            name: name.to_string(),
            qt,
            rows,
            cols,
            offset: 0,
            nbytes,
            route: None,
            density: None,
            subs: Vec::new(),
        }
    }

    #[test]
    fn continue_parameter_splits_rows_without_subs() {
        let t = tensor("blk.0.ffn_gate_exps.weight", QuantType::Q2K, 6, 256);
        let slices = continue_parameter_slices(&t, 4).unwrap();
        assert_eq!(slices.len(), 4);
        assert!(slices.iter().all(|s| s.independent));
        assert_eq!(slices.iter().map(|s| s.nbytes).sum::<u64>(), t.nbytes);
        assert_eq!(
            slices.iter().map(|s| s.rows.unwrap()).sum::<usize>(),
            t.rows
        );
        let mut cursor = 0;
        for (i, s) in slices.iter().enumerate() {
            assert_eq!(s.index, i);
            assert_eq!(s.byte_offset, cursor);
            cursor += s.nbytes;
        }
    }

    #[test]
    fn continue_parameter_respects_blocks() {
        // 10 bytes F32 no se dividen por filas: 10 bloques de 4 bytes.
        let t = TensorInfo {
            name: "irregular".to_string(),
            qt: QuantType::F32,
            rows: 1,
            cols: 10,
            offset: 0,
            nbytes: 10,
            route: None,
            density: None,
            subs: Vec::new(),
        };
        // 10 bytes = 2 bloques de 4 + 2 bytes de resto: no se pueden hacer 3
        // cortes sin romper un bloque, así que salen 2 parámetros.
        let slices = continue_parameter_slices(&t, 3).unwrap();
        assert_eq!(slices.len(), 2);
        assert_eq!(slices.iter().map(|s| s.nbytes).sum::<u64>(), 10);
        assert_eq!(slices[0].nbytes % 4, 0);
        assert_eq!(slices[1].byte_offset, slices[0].nbytes);
    }
}
