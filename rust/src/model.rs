//! Motor de inferencia CPU para OLMoE leído directamente de un `.ukj`.
//!
//! Los pesos no se copian: se mapea el archivo completo y cada tensor se
//! decuantiza por bloques al vuelo. El KV cache es la única memoria que crece
//! con la conversación.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

use memmap2::{Mmap, MmapOptions};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::format::Header;
use crate::operators::{OperatorConfig, OperatorEvent, SessionLog, TicsTracker};
use crate::quant::{self, dequant_row, QuantType};
use crate::reader::UkjFile;
use crate::tokenizer::Tokenizer;

pub const PAGE_SIZE: usize = 4096;

#[derive(Debug, Clone)]
pub struct OlmoeConfig {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub experts: usize,
    pub top_k: usize,
    pub vocab: usize,
    pub eps: f32,
    pub rope_theta: f32,
    pub context: usize,
    inv_freq: Vec<f32>,
}

fn meta_u64(meta: &serde_json::Map<String, Value>, key: &str) -> Result<usize> {
    meta.get(key)
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .ok_or_else(|| Error::new(format!("falta el metadato {key} en la cabecera")))
}

fn meta_f32(meta: &serde_json::Map<String, Value>, key: &str, default: f32) -> f32 {
    meta.get(key)
        .and_then(Value::as_f64)
        .map(|v| v as f32)
        .unwrap_or(default)
}

impl OlmoeConfig {
    pub fn from_header(header: &Header) -> Result<Self> {
        let meta = header
            .model
            .metadata
            .as_object()
            .ok_or_else(|| Error::new("la cabecera no contiene model.metadata"))?;
        let arch = meta
            .get("general.architecture")
            .and_then(Value::as_str)
            .unwrap_or("");
        if arch != "olmoe" {
            return Err(Error::new(format!(
                "arquitectura '{arch}' no soportada por el motor OLMoE"
            )));
        }
        let hidden = meta_u64(meta, "olmoe.embedding_length")?;
        let layers = meta_u64(meta, "olmoe.block_count")?;
        let heads = meta_u64(meta, "olmoe.attention.head_count")?;
        let kv_heads = meta_u64(meta, "olmoe.attention.head_count_kv")?;
        let intermediate = meta_u64(meta, "olmoe.feed_forward_length")?;
        let experts = meta_u64(meta, "olmoe.expert_count")?;
        let top_k = meta_u64(meta, "olmoe.expert_used_count")?;
        let context = meta_u64(meta, "olmoe.context_length")?;
        let vocab = meta
            .get("tokenizer.ggml.tokens")
            .and_then(Value::as_array)
            .map(|a| a.len())
            .unwrap_or(0);
        if vocab == 0 {
            return Err(Error::new("la cabecera no declara el vocabulario"));
        }
        if heads == 0 || hidden % heads != 0 {
            return Err(Error::new("head_count incompatible con hidden_size"));
        }
        if kv_heads == 0 || kv_heads > heads || heads % kv_heads != 0 {
            return Err(Error::new("head_count_kv inválido"));
        }
        let head_dim = hidden / heads;
        let eps = meta_f32(meta, "olmoe.attention.layer_norm_rms_epsilon", 1e-5);
        let rope_theta = meta_f32(meta, "olmoe.rope.freq_base", 10000.0);
        let inv_freq = (0..head_dim / 2)
            .map(|i| 1.0 / rope_theta.powf((2 * i) as f32 / head_dim as f32))
            .collect();
        Ok(Self {
            hidden,
            layers,
            heads,
            kv_heads,
            head_dim,
            intermediate,
            experts,
            top_k,
            vocab,
            eps,
            rope_theta,
            context,
            inv_freq,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SubRange {
    pub index: u32,
    pub row_start: u64,
    pub row_count: u64,
    pub byte_offset: u64,
    pub nbytes: u64,
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub qt: QuantType,
    pub rows: usize,
    pub cols: usize,
    pub offset: u64,
    pub nbytes: u64,
    pub route: Option<String>,
    pub density: Option<String>,
    pub subs: Vec<SubRange>,
}

impl TensorInfo {
    pub fn row_bytes(&self) -> Result<usize> {
        self.qt.row_bytes(self.cols)
    }

    /// Bytes que ocupan `count` filas a partir de `start`.
    pub fn row_range_bytes(&self, start: usize, count: usize) -> Result<u64> {
        if start + count > self.rows {
            return Err(Error::new(format!(
                "tensor '{}': filas {start}..{} fuera de {}",
                self.name,
                start + count,
                self.rows
            )));
        }
        let rb = self.row_bytes()? as u64;
        Ok(start as u64 * rb + count as u64 * rb)
    }

    /// Offset relativo al payload de una fila.
    pub fn row_offset(&self, start: usize) -> Result<u64> {
        let rb = self.row_bytes()? as u64;
        Ok(self.offset + start as u64 * rb)
    }
}

pub struct OlmoeModel {
    pub ukj: UkjFile,
    mmap: Mmap,
    payload_offset: usize,
    cfg: OlmoeConfig,
    tensors: HashMap<String, TensorInfo>,
    /// Tensores lógicos reensamblados desde cortes de `continue parameter`.
    logical: HashMap<String, TensorInfo>,
    pub tokenizer: Tokenizer,
    /// Máscara opcional de expertos permitidos por capa (bit e = experto e).
    expert_mask: Option<Vec<u64>>,
    /// Con máscara, reparte el peso de los expertos permitidos para conservar
    /// la escala que OLMoE no renormaliza.
    renormalize_moe: bool,
    /// Contadores de uso por capa y experto (diagnóstico del router real).
    usage: std::cell::RefCell<Option<Vec<[u64; 64]>>>,
    /// Tics y operador stop-subparameter (opcional).
    tics: RefCell<Option<TicsTracker>>,
    operator_log: RefCell<Option<SessionLog>>,
    tics_actions: Cell<u64>,
    /// Backend Vulkan opcional; None usa la ruta CPU.
    vulkan: RefCell<Option<crate::vulkan::VulkanBackend>>,
}

impl OlmoeModel {
    /// Apertura rápida sin verificar checksums (bench, route, ejemplos).
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_with_verify(path, false, false)
    }

    /// Apertura verificando payload y trailer; es la ruta que usa `chat`.
    pub fn open_verified(path: &Path) -> Result<Self> {
        Self::open_with_verify(path, true, true)
    }

    pub fn open_with_verify(
        path: &Path,
        verify_payload: bool,
        verify_trailer: bool,
    ) -> Result<Self> {
        let ukj = UkjFile::open(path, verify_payload, verify_trailer)?;
        let cfg = OlmoeConfig::from_header(&ukj.header)?;
        let tokenizer = Tokenizer::from_header(&ukj.header)?;
        let file = File::open(path)
            .map_err(|e| Error::new(format!("no se puede abrir {}: {e}", path.display())))?;
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| Error::new(format!("no se puede mapear {}: {e}", path.display())))?
        };
        let payload_offset = ukj.container.payload_offset as usize;
        if payload_offset + ukj.container.payload_len as usize > mmap.len() {
            return Err(Error::new("el payload declarado excede el archivo mapeado"));
        }

        let mut tensors = HashMap::with_capacity(ukj.header.tensors.len());
        // Cortes de `continue parameter`: (nombre base, entrada física).
        let mut continued: HashMap<String, Vec<(u64, u64, String)>> = HashMap::new();
        for t in &ukj.header.tensors {
            let cols = t.dims.first().copied().unwrap_or(0) as usize;
            let rows = if t.dims.len() <= 1 {
                1
            } else {
                t.dims[1..].iter().product::<u64>() as usize
            };
            let qt = QuantType::from_ggml(t.ggml_type)?;
            if t.continued_from.is_none() {
                let expect = qt
                    .row_bytes(cols)?
                    .checked_mul(rows)
                    .ok_or_else(|| Error::new(format!("tensor '{}': tamaño desbordado", t.name)))?;
                if expect != t.nbytes as usize {
                    return Err(Error::new(format!(
                        "tensor '{}': {} bytes no coinciden con {rows}x{cols} ({expect})",
                        t.name, t.nbytes
                    )));
                }
                if let Some(declared) = t.rows {
                    if declared as usize != rows && t.dims.len() > 1 {
                        return Err(Error::new(format!(
                            "tensor '{}': rows={declared} no coincide con las dimensiones ({rows})",
                            t.name
                        )));
                    }
                }
            } else if let Some(declared) = t.rows {
                let expect = qt
                    .row_bytes(cols)?
                    .checked_mul(declared as usize)
                    .ok_or_else(|| Error::new(format!("tensor '{}': tamaño desbordado", t.name)))?;
                if expect != t.nbytes as usize {
                    return Err(Error::new(format!(
                        "corte '{}': {} bytes no coinciden con {declared}x{cols} ({expect})",
                        t.name, t.nbytes
                    )));
                }
            }
            let subs = t
                .subs
                .iter()
                .map(|s| SubRange {
                    index: s.index,
                    row_start: s.row_start,
                    row_count: s.row_count,
                    byte_offset: s.byte_offset,
                    nbytes: s.nbytes,
                })
                .collect();
            if let Some(base) = &t.continued_from {
                continued.entry(base.clone()).or_default().push((
                    t.offset,
                    t.nbytes,
                    t.name.clone(),
                ));
            }
            tensors.insert(
                t.name.clone(),
                TensorInfo {
                    name: t.name.clone(),
                    qt,
                    rows,
                    cols,
                    offset: t.offset,
                    nbytes: t.nbytes,
                    route: t.route.clone(),
                    density: t.density.clone(),
                    subs,
                },
            );
        }

        // Reensambla el tensor lógico para que el motor siga llamándolo por su
        // nombre original aunque el contenedor lo haya partido (definición 46).
        let mut logical = HashMap::new();
        for (base, mut parts) in continued {
            if tensors.contains_key(&base) || parts.is_empty() {
                continue;
            }
            parts.sort_by_key(|p| p.0);
            let first = &tensors[&parts[0].2];
            let mut expected = first.offset;
            let mut total = 0u64;
            for (offset, nbytes, _name) in &parts {
                if *offset != expected {
                    break;
                }
                expected += *nbytes;
                total += *nbytes;
            }
            if total != expected - first.offset {
                continue;
            }
            let Ok(rb) = first.row_bytes() else {
                continue;
            };
            if rb == 0 || total as usize % rb != 0 {
                continue;
            }
            let rows = total as usize / rb;
            logical.insert(
                base.clone(),
                TensorInfo {
                    name: base,
                    qt: first.qt,
                    rows,
                    cols: first.cols,
                    offset: first.offset,
                    nbytes: total,
                    route: None,
                    density: None,
                    subs: Vec::new(),
                },
            );
        }

        Ok(Self {
            ukj,
            mmap,
            payload_offset,
            cfg,
            tensors,
            logical,
            tokenizer,
            expert_mask: None,
            renormalize_moe: false,
            usage: std::cell::RefCell::new(None),
            tics: RefCell::new(None),
            operator_log: RefCell::new(None),
            tics_actions: Cell::new(0),
            vulkan: RefCell::new(None),
        })
    }

    pub fn config(&self) -> &OlmoeConfig {
        &self.cfg
    }

    pub fn tensor_infos(&self) -> &HashMap<String, TensorInfo> {
        &self.tensors
    }

    pub fn tensor_info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .or_else(|| self.logical.get(name))
            .ok_or_else(|| Error::new(format!("tensor '{name}' no encontrado")))
    }

    pub fn payload_offset(&self) -> usize {
        self.payload_offset
    }

    /// Vista del archivo mapeado completo (cabecera incluida).
    pub fn mmap(&self) -> &Mmap {
        &self.mmap
    }

    pub fn payload_len(&self) -> u64 {
        self.ukj.container.payload_len
    }

    pub fn density(&self) -> &str {
        self.ukj
            .header
            .densities
            .declarations
            .as_array()
            .and_then(|a| a.first())
            .and_then(|d| d.get("code"))
            .and_then(Value::as_str)
            .unwrap_or(&self.ukj.header.densities.default)
    }

    /// Activa el backend Vulkan y devuelve la información del adaptador.
    pub fn enable_vulkan(&self, cache_mb: u64) -> Result<crate::vulkan::GpuInfo> {
        let backend = crate::vulkan::VulkanBackend::with_cache_mb(cache_mb)?;
        let info = backend.info().clone();
        *self.vulkan.borrow_mut() = Some(backend);
        Ok(info)
    }

    pub fn disable_vulkan(&self) {
        *self.vulkan.borrow_mut() = None;
    }

    pub fn vulkan_enabled(&self) -> bool {
        self.vulkan.borrow().is_some()
    }

    pub fn vulkan_cached_bytes(&self) -> u64 {
        self.vulkan
            .borrow()
            .as_ref()
            .map(|v| v.cached_bytes())
            .unwrap_or(0)
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let t = self.tensor_info(name)?;
        let start = self.payload_offset + t.offset as usize;
        let end = start + t.nbytes as usize;
        if end > self.mmap.len() {
            return Err(Error::new(format!(
                "tensor '{}' excede el archivo mapeado",
                name
            )));
        }
        Ok(&self.mmap[start..end])
    }

    pub fn tensor_row_bytes(&self, name: &str, start: usize, count: usize) -> Result<&[u8]> {
        let t = self.tensor_info(name)?;
        let rb = t.row_bytes()?;
        if start + count > t.rows {
            return Err(Error::new(format!(
                "tensor '{}': filas {start}..{} fuera de {}",
                name,
                start + count,
                t.rows
            )));
        }
        let off = self.payload_offset + t.offset as usize + start * rb;
        let len = count * rb;
        Ok(&self.mmap[off..off + len])
    }

    /// Restringe el MoE a un subconjunto de expertos (uno por bit). `None`
    /// vuelve a usar los 64 expertos.
    pub fn set_expert_mask(&mut self, mask: Option<Vec<u64>>) -> Result<()> {
        if let Some(m) = &mask {
            if m.len() != self.cfg.layers {
                return Err(Error::new(format!(
                    "máscara de expertos con {} capas; se esperaban {}",
                    m.len(),
                    self.cfg.layers
                )));
            }
        }
        // OLMoE no renormaliza el top-k; hacerlo altera la escala del residual.
        self.renormalize_moe = false;
        self.expert_mask = mask;
        Ok(())
    }

    /// Activa o desactiva la renormalización de pesos MoE.
    pub fn set_renormalize_moe(&mut self, on: bool) {
        self.renormalize_moe = on;
    }

    /// Empieza a contar qué expertos elige el router en cada capa.
    pub fn start_usage(&self) {
        *self.usage.borrow_mut() = Some(vec![[0u64; 64]; self.cfg.layers]);
    }

    /// Devuelve y limpia los contadores de uso.
    pub fn take_usage(&self) -> Option<Vec<[u64; 64]>> {
        self.usage.borrow_mut().take()
    }

    /// Activa los tics y el operador `stop-subparameter`.
    pub fn start_tics(&self, cfg: OperatorConfig, log_path: Option<PathBuf>) -> Result<()> {
        let tracker = TicsTracker::new(cfg, self.cfg.layers)?;
        let log = match log_path {
            Some(path) => Some(SessionLog::create(path)?),
            None => None,
        };
        *self.tics.borrow_mut() = Some(tracker);
        *self.operator_log.borrow_mut() = log;
        self.tics_actions.set(0);
        Ok(())
    }

    /// Devuelve el tracker de tics con sus transiciones.
    pub fn take_tics(&self) -> Option<TicsTracker> {
        self.tics.borrow_mut().take()
    }

    pub fn tics_actions(&self) -> u64 {
        self.tics_actions.get()
    }

    pub fn operator_log_path(&self) -> Option<PathBuf> {
        self.operator_log
            .borrow()
            .as_ref()
            .map(|l| l.path().to_path_buf())
    }

    /// Rangos (inicio, longitud) de los tensores de expertos de una capa.
    fn expert_ranges(&self, layer: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::with_capacity(3);
        for suffix in [
            "ffn_gate_exps.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight",
        ] {
            if let Ok(t) = self.tensor_info(&format!("blk.{layer}.{suffix}")) {
                out.push((self.payload_offset + t.offset as usize, t.nbytes as usize));
            }
        }
        out
    }

    /// Despliega o suelta las páginas de todos los subparámetros (expertos)
    /// de una capa. Devuelve los bytes afectados.
    fn madvise_layer_experts(&self, layer: usize, advice: libc::c_int) -> u64 {
        let page = PAGE_SIZE as usize;
        let mut touched = 0u64;
        for (start, len) in self.expert_ranges(layer) {
            let aligned = start & !(page - 1);
            let end = (start + len + page - 1) & !(page - 1);
            if end <= aligned {
                continue;
            }
            let ptr = unsafe { self.mmap.as_ptr().add(aligned) as *mut libc::c_void };
            let rc = unsafe { libc::madvise(ptr, end - aligned, advice) };
            if rc == 0 {
                touched += (end - aligned) as u64;
            }
        }
        touched
    }

    /// Reevalúa todas las capas sin actividad nueva; así el offload ocurre
    /// también cuando el modelo queda idle entre turnos.
    fn tick_tics(&self) {
        let now = Instant::now();
        let transitions = {
            let mut slot = self.tics.borrow_mut();
            match slot.as_mut() {
                Some(t) => {
                    let mut out = Vec::new();
                    for layer in 0..t.layers() {
                        if let Some(tr) = t.tick(layer, now) {
                            out.push(tr);
                        }
                    }
                    out
                }
                None => Vec::new(),
            }
        };
        for tr in transitions {
            self.apply_transition(tr);
        }
    }

    fn apply_tics(&self, layer: usize, activations: usize) {
        let transition = {
            let mut slot = self.tics.borrow_mut();
            match slot.as_mut() {
                Some(t) => t.record(layer, activations, Instant::now()),
                None => None,
            }
        };
        let Some(tr) = transition else { return };
        self.apply_transition(tr);
    }

    fn apply_transition(&self, tr: crate::operators::TicsTransition) {
        let layer = tr.layer;
        let advice = match tr.to {
            crate::operators::DeploymentState::Deployed => libc::MADV_WILLNEED,
            crate::operators::DeploymentState::Stopped => libc::MADV_DONTNEED,
            crate::operators::DeploymentState::Holding => 0,
        };
        let bytes = if advice != 0 {
            self.madvise_layer_experts(layer, advice)
        } else {
            0
        };
        self.tics_actions.set(self.tics_actions.get() + 1);
        if let Some(log) = self.operator_log.borrow().as_ref() {
            let event = OperatorEvent {
                ts_ms: tr.ts_ms,
                operator: "stop-subparameter".to_string(),
                target: format!("blk.{layer}.ffn_*_exps"),
                tics_per_s: tr.tics_per_s,
                state: tr.to.as_str().to_string(),
                action: format!("{} · {} bytes", tr.action, bytes),
                rss_bytes: rss_bytes(),
            };
            let _ = log.write(&event);
        }
    }

    pub fn expert_mask(&self) -> Option<&[u64]> {
        self.expert_mask.as_deref()
    }

    fn norm_weight(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.tensor_info(name)?;
        if t.rows != 1 {
            return Err(Error::new(format!("'{name}' no es un vector de norma")));
        }
        let mut out = vec![0.0f32; t.cols];
        dequant_row(t.qt, self.tensor_bytes(name)?, t.cols, &mut out)?;
        Ok(out)
    }

    fn rmsnorm(&self, x: &[f32], name: &str) -> Result<Vec<f32>> {
        let w = self.norm_weight(name)?;
        let mut out = x.to_vec();
        rmsnorm_inplace(&mut out, &w, self.cfg.eps);
        Ok(out)
    }

    fn matvec(&self, name: &str, x: &[f32], out: &mut [f32]) -> Result<()> {
        let t = self.tensor_info(name)?;
        if x.len() != t.cols {
            return Err(Error::new(format!(
                "'{name}': entrada de {} valores; se esperaban {}",
                x.len(),
                t.cols
            )));
        }
        if out.len() != t.rows {
            return Err(Error::new(format!(
                "'{name}': salida de {} valores; se esperaban {}",
                out.len(),
                t.rows
            )));
        }
        let bytes = self.tensor_bytes(name)?;
        if let Some(gpu) = self.vulkan.borrow_mut().as_mut() {
            return gpu.matvec(name, t.qt, bytes, 0, t.rows, t.cols, x, out);
        }
        quant::matvec(t.qt, bytes, t.rows, t.cols, x, out)
    }

    /// Matvec sobre un rango de bytes del tensor (una fila o un experto).
    fn matvec_slice(
        &self,
        name: &str,
        byte_offset: usize,
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        let t = self.tensor_info(name)?;
        let row_bytes = t.row_bytes()?;
        let len = rows
            .checked_mul(row_bytes)
            .ok_or_else(|| Error::new("matvec_slice: tamaño desbordado"))?;
        if byte_offset + len > t.nbytes as usize {
            return Err(Error::new(format!(
                "tensor '{name}': rango {byte_offset}..{} fuera de {} bytes",
                byte_offset + len,
                t.nbytes
            )));
        }
        let bytes = self.tensor_bytes(name)?;
        if let Some(gpu) = self.vulkan.borrow_mut().as_mut() {
            return gpu.matvec(name, t.qt, bytes, byte_offset, rows, cols, x, out);
        }
        quant::matvec(
            t.qt,
            &bytes[byte_offset..byte_offset + len],
            rows,
            cols,
            x,
            out,
        )
    }

    fn embedding(&self, token: u32) -> Result<Vec<f32>> {
        let t = self.tensor_info("token_embd.weight")?;
        if token as usize >= t.rows {
            return Err(Error::new(format!(
                "token {token} fuera del vocabulario ({})",
                t.rows
            )));
        }
        let rb = t.row_bytes()?;
        let off = self.payload_offset + t.offset as usize + token as usize * rb;
        let mut out = vec![0.0f32; t.cols];
        dequant_row(t.qt, &self.mmap[off..off + rb], t.cols, &mut out)?;
        Ok(out)
    }

    /// Prefill: procesa todos los tokens y devuelve los logits del último.
    pub fn forward_tokens(&mut self, tokens: &[u32], cache: &mut KvCache) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(Error::new("no hay tokens que procesar"));
        }
        if cache.len + tokens.len() > cache.capacity {
            return Err(Error::new(format!(
                "contexto lleno ({} tokens); usa --ctx para ampliarlo",
                cache.capacity
            )));
        }
        let mut logits = Vec::new();
        for &token in tokens {
            logits = self.forward_token(token, cache)?;
        }
        Ok(logits)
    }

    pub fn forward_token(&mut self, token: u32, cache: &mut KvCache) -> Result<Vec<f32>> {
        self.tick_tics();
        if cache.len >= cache.capacity {
            return Err(Error::new(format!(
                "contexto lleno ({} tokens); usa --ctx para ampliarlo",
                cache.capacity
            )));
        }
        let pos = cache.len;
        let mut h = self.embedding(token)?;
        for layer in 0..self.cfg.layers {
            h = self.layer_forward(layer, &h, pos, cache)?;
        }
        let hn = self.rmsnorm(&h, "output_norm.weight")?;
        let mut logits = vec![0.0f32; self.cfg.vocab];
        self.matvec("output.weight", &hn, &mut logits)?;
        cache.len += 1;
        Ok(logits)
    }

    fn layer_forward(
        &self,
        layer: usize,
        h: &[f32],
        pos: usize,
        cache: &mut KvCache,
    ) -> Result<Vec<f32>> {
        let p = |s: &str| format!("blk.{layer}.{s}");
        let hidden = self.cfg.hidden;
        let hd = self.cfg.head_dim;
        let kv_dim = self.cfg.kv_heads * hd;
        let hn = self.rmsnorm(h, &p("attn_norm.weight"))?;

        let mut q = vec![0.0f32; hidden];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        self.matvec(&p("attn_q.weight"), &hn, &mut q)?;
        self.matvec(&p("attn_k.weight"), &hn, &mut k)?;
        self.matvec(&p("attn_v.weight"), &hn, &mut v)?;

        // OLMoE aplica q_norm/k_norm sobre la proyección completa, antes de
        // dividir en cabezas. En GQA la norma de K cubre solo kv_heads.
        let qn = self.norm_weight(&p("attn_q_norm.weight"))?;
        let kn = self.norm_weight(&p("attn_k_norm.weight"))?;
        if qn.len() != q.len() || kn.len() != k.len() {
            return Err(Error::new(format!(
                "capa {layer}: normas q/k incompatibles con GQA ({} y {} vs {} y {})",
                qn.len(),
                kn.len(),
                q.len(),
                k.len()
            )));
        }
        rmsnorm_inplace(&mut q, &qn, self.cfg.eps);
        rmsnorm_inplace(&mut k, &kn, self.cfg.eps);

        apply_rope(&mut q, pos, self.cfg.heads, &self.cfg);
        apply_rope(&mut k, pos, self.cfg.kv_heads, &self.cfg);
        cache.append(layer, &k, &v, pos);

        let attn = attention_heads(&q, cache, layer, pos, self.cfg.heads, self.cfg.kv_heads, hd);

        let mut ao = vec![0.0f32; hidden];
        self.matvec(&p("attn_output.weight"), &attn, &mut ao)?;
        let mut h1 = vec![0.0f32; hidden];
        for i in 0..hidden {
            h1[i] = h[i] + ao[i];
        }

        let hn2 = self.rmsnorm(&h1, &p("ffn_norm.weight"))?;
        let moe = self.moe_forward(layer, &hn2)?;
        let mut out = vec![0.0f32; hidden];
        for i in 0..hidden {
            out[i] = h1[i] + moe[i];
        }
        Ok(out)
    }

    fn moe_forward(&self, layer: usize, x: &[f32]) -> Result<Vec<f32>> {
        let p = |s: &str| format!("blk.{layer}.{s}");
        let hidden = self.cfg.hidden;
        let inter = self.cfg.intermediate;

        let mut probs = vec![0.0f32; self.cfg.experts];
        self.matvec(&p("ffn_gate_inp.weight"), x, &mut probs)?;
        softmax_inplace(&mut probs);

        let mut order: Vec<usize> = (0..self.cfg.experts).collect();
        if let Some(mask) = &self.expert_mask {
            let bits = mask[layer];
            order.retain(|&e| bits & (1u64 << e) != 0);
            if order.is_empty() {
                return Err(Error::new(format!(
                    "capa {layer}: la máscara de expertos no deja ninguno"
                )));
            }
        }
        order.sort_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let k = self.cfg.top_k.min(order.len());

        if let Some(usage) = self.usage.borrow_mut().as_mut() {
            for &e in &order[..k] {
                usage[layer][e] += 1;
            }
        }

        self.apply_tics(layer, k);

        let mut weights: Vec<f32> = order[..k].iter().map(|&e| probs[e]).collect();
        if self.renormalize_moe {
            let sum: f32 = weights.iter().sum();
            if sum > 0.0 {
                for w in weights.iter_mut() {
                    *w /= sum;
                }
            }
        }

        let mut out = vec![0.0f32; hidden];
        for (slot, &expert) in order[..k].iter().enumerate() {
            let w = weights[slot];
            let gate = self.expert_matvec(&p("ffn_gate_exps.weight"), expert, inter, hidden, x)?;
            let up = self.expert_matvec(&p("ffn_up_exps.weight"), expert, inter, hidden, x)?;
            let act: Vec<f32> = gate.iter().zip(&up).map(|(&g, &u)| silu(g) * u).collect();
            let down =
                self.expert_matvec(&p("ffn_down_exps.weight"), expert, hidden, inter, &act)?;
            for i in 0..hidden {
                out[i] += w * down[i];
            }
        }
        Ok(out)
    }

    fn expert_matvec(
        &self,
        name: &str,
        expert: usize,
        rows_per_expert: usize,
        cols: usize,
        x: &[f32],
    ) -> Result<Vec<f32>> {
        let t = self.tensor_info(name)?;
        if rows_per_expert == 0 || t.rows % rows_per_expert != 0 {
            return Err(Error::new(format!(
                "tensor '{}': filas no divisibles por experto",
                t.name
            )));
        }
        let rb = t.row_bytes()?;
        let start = expert * rows_per_expert * rb;
        let end = start + rows_per_expert * rb;
        if end > t.nbytes as usize {
            return Err(Error::new(format!(
                "tensor '{}': experto {expert} fuera de rango",
                t.name
            )));
        }
        let mut out = vec![0.0f32; rows_per_expert];
        self.matvec_slice(name, start, rows_per_expert, cols, x, &mut out)?;
        Ok(out)
    }
}

pub struct KvCache {
    capacity: usize,
    pub len: usize,
    kv_heads: usize,
    head_dim: usize,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl KvCache {
    pub fn new(layers: usize, kv_heads: usize, head_dim: usize, capacity: usize) -> Self {
        Self {
            capacity,
            len: 0,
            kv_heads,
            head_dim,
            k: (0..layers).map(|_| Vec::new()).collect(),
            v: (0..layers).map(|_| Vec::new()).collect(),
        }
    }

    fn ensure(&mut self, layer: usize) {
        if self.k[layer].is_empty() {
            let n = self.kv_heads * self.capacity * self.head_dim;
            self.k[layer] = vec![0.0f32; n];
            self.v[layer] = vec![0.0f32; n];
        }
    }

    fn append(&mut self, layer: usize, k: &[f32], v: &[f32], pos: usize) {
        self.ensure(layer);
        for h in 0..self.kv_heads {
            let dst = (h * self.capacity + pos) * self.head_dim;
            let src = h * self.head_dim;
            self.k[layer][dst..dst + self.head_dim].copy_from_slice(&k[src..src + self.head_dim]);
            self.v[layer][dst..dst + self.head_dim].copy_from_slice(&v[src..src + self.head_dim]);
        }
    }

    fn k_at(&self, layer: usize, head: usize, t: usize) -> &[f32] {
        let base = (head * self.capacity + t) * self.head_dim;
        &self.k[layer][base..base + self.head_dim]
    }

    fn v_at(&self, layer: usize, head: usize, t: usize) -> &[f32] {
        let base = (head * self.capacity + t) * self.head_dim;
        &self.v[layer][base..base + self.head_dim]
    }

    pub fn reset(&mut self) {
        self.len = 0;
        for v in &mut self.k {
            v.clear();
        }
        for v in &mut self.v {
            v.clear();
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes reservados actualmente para K y V.
    pub fn bytes(&self) -> usize {
        self.k.iter().map(Vec::len).sum::<usize>() * 4 * 2
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    let mut i = 0;
    while i + 4 <= a.len() {
        acc += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
        i += 4;
    }
    while i < a.len() {
        acc += a[i] * b[i];
        i += 1;
    }
    acc
}

fn rmsnorm_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let n = x.len();
    let mut ss = 0.0f32;
    for &v in x.iter() {
        ss += v * v;
    }
    let scale = 1.0 / (ss / n as f32 + eps).sqrt();
    for i in 0..n {
        x[i] = x[i] * scale * w[i];
    }
}

fn softmax_inplace(x: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Atención multi-cabeza con GQA: cada cabeza Q usa la cabeza KV
/// `head % kv_heads`, de modo que varias cabezas Q comparten una KV.
fn attention_heads(
    q: &[f32],
    cache: &KvCache,
    layer: usize,
    pos: usize,
    heads: usize,
    kv_heads: usize,
    hd: usize,
) -> Vec<f32> {
    let mut attn = vec![0.0f32; heads * hd];
    let scale = 1.0 / (hd as f32).sqrt();
    for head in 0..heads {
        let kv_head = head % kv_heads;
        let qh = &q[head * hd..(head + 1) * hd];
        let mut scores = vec![0.0f32; pos + 1];
        for (t, score) in scores.iter_mut().enumerate() {
            let kh = cache.k_at(layer, kv_head, t);
            *score = dot(qh, kh) * scale;
        }
        softmax_inplace(&mut scores);
        let out = &mut attn[head * hd..(head + 1) * hd];
        for (t, &w) in scores.iter().enumerate() {
            let vh = cache.v_at(layer, kv_head, t);
            for d in 0..hd {
                out[d] += w * vh[d];
            }
        }
    }
    attn
}

fn apply_rope(x: &mut [f32], pos: usize, heads: usize, cfg: &OlmoeConfig) {
    let hd = cfg.head_dim;
    for head in 0..heads {
        let base = head * hd;
        for i in 0..hd / 2 {
            let angle = pos as f32 * cfg.inv_freq[i];
            let (sin, cos) = angle.sin_cos();
            let a = x[base + i];
            let b = x[base + i + hd / 2];
            x[base + i] = a * cos - b * sin;
            x[base + i + hd / 2] = b * cos + a * sin;
        }
    }
}

/// Memoria residente actual del proceso (VmRSS).
pub fn rss_bytes() -> u64 {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            if let Some(kb) = rest.split_whitespace().next() {
                if let Ok(v) = kb.parse::<u64>() {
                    return v * 1024;
                }
            }
        }
    }
    0
}

pub fn peak_rss_bytes() -> u64 {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            if let Some(kb) = rest.split_whitespace().next() {
                if let Ok(v) = kb.parse::<u64>() {
                    return v * 1024;
                }
            }
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_one() {
        let mut v = [1.0, 2.0, 3.0, -1.0];
        softmax_inplace(&mut v);
        let s: f32 = v.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
        assert!(v[2] > v[1] && v[1] > v[0] && v[0] > v[3]);
    }

    #[test]
    fn rmsnorm_matches_formula() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let w = [1.0f32, 1.0, 1.0, 1.0];
        let mut y = x;
        rmsnorm_inplace(&mut y, &w, 1e-5);
        let ss: f32 = x.iter().map(|v| v * v).sum();
        let scale = 1.0 / (ss / 4.0 + 1e-5).sqrt();
        for i in 0..4 {
            assert!((y[i] - x[i] * scale).abs() < 1e-6);
        }
    }

    #[test]
    fn rope_preserves_norm() {
        let cfg = OlmoeConfig {
            hidden: 8,
            layers: 1,
            heads: 1,
            kv_heads: 1,
            head_dim: 8,
            intermediate: 4,
            experts: 2,
            top_k: 1,
            vocab: 10,
            eps: 1e-5,
            rope_theta: 10000.0,
            context: 16,
            inv_freq: (0..4)
                .map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / 8.0))
                .collect(),
        };
        let mut x = [0.3f32, -0.7, 1.1, 0.2, 0.5, -0.1, 0.9, -0.4];
        let before: f32 = x.iter().map(|v| v * v).sum();
        apply_rope(&mut x, 7, cfg.heads, &cfg);
        let after: f32 = x.iter().map(|v| v * v).sum();
        assert!((before - after).abs() < 1e-5);
    }

    #[test]
    fn gqa_heads_share_kv_cache() {
        let heads = 4;
        let kv_heads = 2;
        let hd = 2;
        let mut cache = KvCache::new(1, kv_heads, hd, 2);
        // K en cero => softmax uniforme; V identifica la cabeza KV usada.
        let k = vec![0.0f32; kv_heads * hd];
        let v = [1.0f32, 1.0, 2.0, 2.0];
        cache.append(0, &k, &v, 0);

        let q = vec![0.0f32; heads * hd];
        let out = attention_heads(&q, &cache, 0, 0, heads, kv_heads, hd);
        for head in 0..heads {
            let expected = if head % kv_heads == 0 { 1.0 } else { 2.0 };
            for d in 0..hd {
                assert!(
                    (out[head * hd + d] - expected).abs() < 1e-6,
                    "cabeza {head} dimensión {d}: {} != {expected}",
                    out[head * hd + d]
                );
            }
        }
    }
}
