//! Plan de rutas y carga perezosa: solo se tocan las páginas activadas.
//!
//! Dos modos:
//! - **tabla de rutas**: cada mundo referencia rutas del `.ukj`; se cargan sus
//!   tensores completos.
//! - **expertos**: cada mundo referencia expertos del MoE; se cargan solo las
//!   filas de esos expertos, más la parte no experta de cada capa.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::time::Instant;

use serde::Serialize;

use crate::error::{Error, Result};
use crate::format::Route;
use crate::model::{
    madvise_range, rss_bytes, OlmoeModel, TensorInfo, MADV_DONTNEED, MADV_WILLNEED, PAGE_SIZE,
};
use crate::router::{Activation, WorldTaxonomy};

#[derive(Debug, Clone, Serialize)]
pub struct ByteRange {
    pub offset: u64,
    pub len: u64,
    pub label: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct RoutePlan {
    pub worlds: Vec<crate::router::ActivatedWorld>,
    pub routes: Vec<String>,
    pub tensors: Vec<String>,
    pub expert_mask: Option<Vec<u64>>,
    pub ranges: Vec<ByteRange>,
    pub loaded_bytes: u64,
    pub full_bytes: u64,
    pub incomplete: bool,
    pub warning: Option<String>,
}

impl RoutePlan {
    pub fn ratio(&self) -> f64 {
        if self.full_bytes == 0 {
            0.0
        } else {
            self.loaded_bytes as f64 / self.full_bytes as f64
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct LoadStats {
    pub bytes: u64,
    pub pages: u64,
    pub rss_before: u64,
    pub rss_after: u64,
    /// Diferencia con signo: positiva al cargar, negativa al liberar.
    pub rss_delta: i64,
    pub elapsed_ms: f64,
}

fn expert_row_counts(name: &str, info: &TensorInfo, n_experts: usize) -> Option<(usize, usize)> {
    if !name.ends_with("ffn_gate_exps.weight")
        && !name.ends_with("ffn_up_exps.weight")
        && !name.ends_with("ffn_down_exps.weight")
    {
        return None;
    }
    if n_experts == 0 || info.rows % n_experts != 0 {
        return None;
    }
    Some((info.rows / n_experts, n_experts))
}

fn push_range(
    ranges: &mut Vec<ByteRange>,
    seen: &mut HashSet<(u64, u64)>,
    offset: u64,
    len: u64,
    label: String,
) {
    if len == 0 || !seen.insert((offset, len)) {
        return;
    }
    ranges.push(ByteRange { offset, len, label });
}

/// Máscara de expertos por capa para uno o varios mundos. Cada bit `e`
/// habilita el experto `e` en todas las capas.
pub fn expert_mask_for(
    taxonomy: &WorldTaxonomy,
    world_ids: &[String],
    layers: usize,
    n_experts: usize,
) -> Result<Vec<u64>> {
    let mut mask = vec![0u64; layers];
    let mut any = false;
    for id in world_ids {
        let spec = taxonomy
            .worlds
            .iter()
            .find(|s| &s.id == id)
            .ok_or_else(|| Error::new(format!("mundo '{id}' no está en la taxonomía")))?;
        if !spec.experts_by_layer.is_empty() {
            if spec.experts_by_layer.len() != layers {
                return Err(Error::new(format!(
                    "el mundo '{}' declara {} capas de expertos; el modelo tiene {layers}",
                    spec.id,
                    spec.experts_by_layer.len()
                )));
            }
            for (layer, list) in spec.experts_by_layer.iter().enumerate() {
                for &e in list {
                    if e as usize >= n_experts {
                        return Err(Error::new(format!(
                            "el mundo '{}' referencia el experto {e} en la capa {layer}; el modelo tiene {n_experts}",
                            spec.id
                        )));
                    }
                    mask[layer] |= 1u64 << e;
                    any = true;
                }
            }
            continue;
        }
        for &e in &spec.experts {
            if e as usize >= n_experts {
                return Err(Error::new(format!(
                    "el mundo '{}' referencia el experto {e}; el modelo tiene {n_experts}",
                    spec.id
                )));
            }
            for layer in 0..layers {
                mask[layer] |= 1u64 << e;
            }
            any = true;
        }
    }
    if !any {
        return Err(Error::new(
            "los mundos indicados no declaran expertos utilizables",
        ));
    }
    Ok(mask)
}

/// Construye el plan de carga para una activación del router.
pub fn build_plan(
    taxonomy: &WorldTaxonomy,
    activation: &Activation,
    routes: &[Route],
    tensors: &HashMap<String, TensorInfo>,
    n_experts: usize,
    full_bytes: u64,
) -> Result<RoutePlan> {
    let route_by_id: HashMap<&str, &Route> = routes.iter().map(|r| (r.id.as_str(), r)).collect();
    let mut ranges = Vec::new();
    let mut seen = HashSet::new();
    let mut tensors_used = BTreeSet::new();
    let mut routes_used = BTreeSet::new();
    let mut warning = activation.warning.clone();

    if activation.incomplete {
        // Fallback explícito: modo incompleto con el modelo base completo.
        warning = Some(match warning {
            Some(w) => format!("{w}; se carga el modelo base completo"),
            None => "modo incompleto: se carga el modelo base completo".to_string(),
        });
        for r in routes {
            routes_used.insert(r.id.clone());
            for name in &r.tensors {
                let info = tensors.get(name).ok_or_else(|| {
                    Error::new(format!(
                        "la ruta '{}' referencia el tensor '{name}' inexistente",
                        r.id
                    ))
                })?;
                push_range(
                    &mut ranges,
                    &mut seen,
                    info.offset,
                    info.nbytes,
                    name.clone(),
                );
                tensors_used.insert(name.clone());
            }
        }
        let loaded_bytes = ranges.iter().map(|r| r.len).sum();
        return Ok(RoutePlan {
            worlds: Vec::new(),
            routes: routes_used.into_iter().collect(),
            tensors: tensors_used.into_iter().collect(),
            expert_mask: None,
            ranges,
            loaded_bytes,
            full_bytes,
            incomplete: true,
            warning,
        });
    }

    let use_experts = activation.worlds.iter().any(|w| {
        taxonomy
            .worlds
            .iter()
            .find(|s| s.id == w.id)
            .map(|s| !s.experts.is_empty() || !s.experts_by_layer.is_empty())
            .unwrap_or(false)
    });

    if use_experts {
        // El número de capas se deduce de los tensores `blk.N.` presentes.
        let mut layers = BTreeSet::new();
        for name in tensors.keys() {
            if let Some(rest) = name.strip_prefix("blk.") {
                if let Some((n, _)) = rest.split_once('.') {
                    if let Ok(v) = n.parse::<usize>() {
                        layers.insert(v);
                    }
                }
            }
        }
        let n_layers = layers.len();
        let world_ids: Vec<String> = activation.worlds.iter().map(|w| w.id.clone()).collect();
        let expert_mask = expert_mask_for(taxonomy, &world_ids, n_layers, n_experts)?;

        // Globales + todo lo no experto de cada capa + filas de expertos.
        let mut sorted_names: Vec<&String> = tensors.keys().collect();
        sorted_names.sort();
        for name in sorted_names {
            let info = &tensors[name];
            if name.starts_with("blk.") {
                if let Some((rows_per_expert, n)) = expert_row_counts(name, info, n_experts) {
                    let rb = info.row_bytes()? as u64;
                    let layer = name
                        .strip_prefix("blk.")
                        .and_then(|r| r.split_once('.'))
                        .and_then(|(n, _)| n.parse::<usize>().ok())
                        .unwrap_or(0);
                    let bits = expert_mask.get(layer).copied().unwrap_or(0);
                    for e in 0..n {
                        if bits & (1u64 << e) == 0 {
                            continue;
                        }
                        let off = info.offset + e as u64 * rows_per_expert as u64 * rb;
                        let len = rows_per_expert as u64 * rb;
                        push_range(
                            &mut ranges,
                            &mut seen,
                            off,
                            len,
                            format!("{name}[experto {e}]"),
                        );
                        tensors_used.insert(name.clone());
                    }
                    continue;
                }
            }
            push_range(
                &mut ranges,
                &mut seen,
                info.offset,
                info.nbytes,
                name.clone(),
            );
            tensors_used.insert(name.clone());
        }
        for r in routes {
            routes_used.insert(r.id.clone());
        }
        let loaded_bytes = ranges.iter().map(|r| r.len).sum();
        return Ok(RoutePlan {
            worlds: activation.worlds.clone(),
            routes: routes_used.into_iter().collect(),
            tensors: tensors_used.into_iter().collect(),
            expert_mask: Some(expert_mask),
            ranges,
            loaded_bytes,
            full_bytes,
            incomplete: false,
            warning,
        });
    }

    // Modo tabla de rutas.
    routes_used.insert("global".to_string());
    for w in &activation.worlds {
        let spec = taxonomy
            .worlds
            .iter()
            .find(|s| s.id == w.id)
            .ok_or_else(|| Error::new(format!("mundo '{}' no está en la taxonomía", w.id)))?;
        if spec.routes.is_empty() {
            return Err(Error::new(format!(
                "el mundo '{}' no declara rutas ni expertos",
                w.id
            )));
        }
        for rid in &spec.routes {
            routes_used.insert(rid.clone());
        }
    }
    for rid in &routes_used {
        let route = route_by_id
            .get(rid.as_str())
            .ok_or_else(|| Error::new(format!("la ruta '{rid}' no existe en el .ukj")))?;
        for name in &route.tensors {
            let info = tensors.get(name).ok_or_else(|| {
                Error::new(format!(
                    "la ruta '{rid}' referencia el tensor '{name}' inexistente"
                ))
            })?;
            push_range(
                &mut ranges,
                &mut seen,
                info.offset,
                info.nbytes,
                name.clone(),
            );
            tensors_used.insert(name.clone());
        }
    }
    ranges.sort_by_key(|r| r.offset);
    let loaded_bytes = ranges.iter().map(|r| r.len).sum();
    Ok(RoutePlan {
        worlds: activation.worlds.clone(),
        routes: routes_used.into_iter().collect(),
        tensors: tensors_used.into_iter().collect(),
        expert_mask: None,
        ranges,
        loaded_bytes,
        full_bytes,
        incomplete: false,
        warning,
    })
}

impl OlmoeModel {
    /// Fuerza la residencia de las páginas del plan y mide el RSS.
    pub fn load_plan(&self, plan: &RoutePlan) -> LoadStats {
        let before = rss_bytes();
        let t0 = Instant::now();
        let mut pages = 0u64;
        let base = self.mmap().as_ptr() as usize;
        for r in &plan.ranges {
            let addr = base + self.payload_offset() + r.offset as usize;
            let end = addr + r.len as usize;
            let aligned = addr & !(PAGE_SIZE - 1);
            unsafe {
                madvise_range(
                    aligned as *mut std::ffi::c_void,
                    end - aligned,
                    MADV_WILLNEED,
                );
            }
            let mut p = aligned;
            while p < end {
                // Lectura volátil: impide que el compilador elimine el toque.
                unsafe {
                    std::ptr::read_volatile(p as *const u8);
                }
                p += PAGE_SIZE;
                pages += 1;
            }
        }
        let after = rss_bytes();
        LoadStats {
            bytes: plan.loaded_bytes,
            pages,
            rss_before: before,
            rss_after: after,
            rss_delta: after as i64 - before as i64,
            elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
        }
    }

    /// Libera las páginas del plan (offload en caliente).
    pub fn unload_plan(&self, plan: &RoutePlan) -> LoadStats {
        let before = rss_bytes();
        let t0 = Instant::now();
        let base = self.mmap().as_ptr() as usize;
        for r in &plan.ranges {
            let addr = base + self.payload_offset() + r.offset as usize;
            let end = addr + r.len as usize;
            let aligned = addr & !(PAGE_SIZE - 1);
            unsafe {
                madvise_range(
                    aligned as *mut std::ffi::c_void,
                    end - aligned,
                    MADV_DONTNEED,
                );
            }
        }
        let after = rss_bytes();
        LoadStats {
            bytes: plan.loaded_bytes,
            pages: 0,
            rss_before: before,
            rss_after: after,
            rss_delta: after as i64 - before as i64,
            elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::QuantType;
    use crate::router::{ActivatedWorld, Activation, WorldSpec};

    fn info(name: &str, offset: u64, nbytes: u64) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            qt: QuantType::F32,
            rows: 1,
            cols: nbytes as usize / 4,
            offset,
            nbytes,
            route: None,
            density: None,
            subs: Vec::new(),
        }
    }

    fn fixture() -> (
        WorldTaxonomy,
        Vec<Route>,
        HashMap<String, TensorInfo>,
        Activation,
    ) {
        let mut tensors = HashMap::new();
        tensors.insert("global.w".into(), info("global.w", 0, 1_000_000));
        tensors.insert("l0.w".into(), info("l0.w", 1_000_000, 2_000_000));
        tensors.insert("l1.w".into(), info("l1.w", 3_000_000, 2_000_000));
        tensors.insert("l2.w".into(), info("l2.w", 5_000_000, 2_000_000));
        let routes = vec![
            Route {
                id: "global".into(),
                label: Some("Global".into()),
                kind: Some("global".into()),
                auto: true,
                tensors: vec!["global.w".into()],
            },
            Route {
                id: "capa.0".into(),
                label: Some("Capa 0".into()),
                kind: Some("layer".into()),
                auto: true,
                tensors: vec!["l0.w".into()],
            },
            Route {
                id: "capa.1".into(),
                label: Some("Capa 1".into()),
                kind: Some("layer".into()),
                auto: true,
                tensors: vec!["l1.w".into()],
            },
            Route {
                id: "capa.2".into(),
                label: Some("Capa 2".into()),
                kind: Some("layer".into()),
                auto: true,
                tensors: vec!["l2.w".into()],
            },
        ];
        let taxonomy = WorldTaxonomy {
            version: 1,
            feature_dim: 64,
            worlds: vec![
                WorldSpec {
                    id: "codigo".into(),
                    label: Some("Código".into()),
                    priority: 10,
                    provisional: false,
                    examples: vec!["python".into()],
                    routes: vec!["capa.0".into()],
                    experts: vec![],
                    experts_by_layer: vec![],
                },
                WorldSpec {
                    id: "matematicas".into(),
                    label: Some("Matemáticas".into()),
                    priority: 9,
                    provisional: false,
                    examples: vec!["integral".into()],
                    routes: vec!["capa.1".into()],
                    experts: vec![],
                    experts_by_layer: vec![],
                },
            ],
        };
        let activation = Activation {
            worlds: vec![ActivatedWorld {
                id: "codigo".into(),
                label: "Código".into(),
                priority: 10,
                provisional: false,
                probability: 0.9,
            }],
            incomplete: false,
            warning: None,
            probabilities: vec![("codigo".into(), 0.9), ("matematicas".into(), 0.1)],
        };
        (taxonomy, routes, tensors, activation)
    }

    #[test]
    fn route_plan_loads_only_activated_routes() {
        let (tax, routes, tensors, act) = fixture();
        let plan = build_plan(&tax, &act, &routes, &tensors, 64, 7_000_000).unwrap();
        assert_eq!(
            plan.routes,
            vec!["capa.0".to_string(), "global".to_string()]
        );
        assert_eq!(plan.loaded_bytes, 3_000_000);
        assert!(plan.loaded_bytes < plan.full_bytes);
        assert_eq!(plan.ranges.len(), 2);
    }

    #[test]
    fn incomplete_plan_has_warning_and_full_base() {
        let (tax, routes, tensors, mut act) = fixture();
        act.incomplete = true;
        act.worlds.clear();
        act.warning = Some("mundo no encontrado".into());
        let plan = build_plan(&tax, &act, &routes, &tensors, 64, 7_000_000).unwrap();
        assert!(plan.incomplete);
        assert!(plan.warning.unwrap().contains("modelo base completo"));
        assert_eq!(plan.loaded_bytes, 7_000_000);
    }

    #[test]
    fn expert_plan_loads_a_fraction() {
        // 2 capas, 4 expertos, gate/up 4 filas por experto, down 8.
        let mut tensors = HashMap::new();
        tensors.insert(
            "token_embd.weight".into(),
            info("token_embd.weight", 0, 400_000),
        );
        for layer in 0..2 {
            let base = 1_000_000 + layer * 1_000_000;
            tensors.insert(
                format!("blk.{layer}.attn_q.weight"),
                info(&format!("blk.{layer}.attn_q.weight"), base, 100_000),
            );
            for (kind, rows, cols) in [("gate", 16usize, 4usize), ("up", 16, 4), ("down", 32, 8)] {
                let mut t = info(
                    &format!("blk.{layer}.ffn_{kind}_exps.weight"),
                    base + 100_000 + rows as u64 * 100,
                    rows as u64 * cols as u64 * 4,
                );
                t.rows = rows;
                t.cols = cols;
                t.nbytes = rows as u64 * cols as u64 * 4;
                tensors.insert(t.name.clone(), t);
            }
        }
        let routes = vec![];
        let taxonomy = WorldTaxonomy {
            version: 1,
            feature_dim: 64,
            worlds: vec![WorldSpec {
                id: "codigo".into(),
                label: Some("Código".into()),
                priority: 10,
                provisional: false,
                examples: vec!["python".into()],
                routes: vec![],
                experts: vec![0, 1],
                experts_by_layer: vec![],
            }],
        };
        let act = Activation {
            worlds: vec![ActivatedWorld {
                id: "codigo".into(),
                label: "Código".into(),
                priority: 10,
                provisional: false,
                probability: 0.9,
            }],
            incomplete: false,
            warning: None,
            probabilities: vec![("codigo".into(), 0.9)],
        };
        let full: u64 = tensors.values().map(|t| t.nbytes).sum();
        let plan = build_plan(&taxonomy, &act, &routes, &tensors, 4, full).unwrap();
        assert_eq!(plan.expert_mask.as_ref().unwrap().len(), 2);
        assert_eq!(plan.expert_mask.as_ref().unwrap()[0], 0b11);
        assert!(plan.loaded_bytes < full);
        // Los globales y attn_q se cargan enteros; de los expertos, solo 2 de 4.
        assert!(plan.tensors.iter().any(|t| t == "token_embd.weight"));
        assert!(plan.tensors.iter().any(|t| t.contains("ffn_gate_exps")));
    }
}
