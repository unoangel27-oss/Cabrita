//! Router tiny de mundos: clasificador lineal sobre n-gramas hasheados.
//!
//! Se entrena en milisegundos con los ejemplos etiquetados de la taxonomía.
//! Devuelve una o varias rutas activadas, con prioridad y probabilidad, o el
//! fallback a modo incompleto con aviso.

use std::collections::BTreeMap;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldSpec {
    pub id: String,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub priority: i32,
    #[serde(default)]
    pub provisional: bool,
    #[serde(default)]
    pub examples: Vec<String>,
    /// Rutas del `.ukj` que pertenecen a este mundo (modo tabla de rutas).
    #[serde(default)]
    pub routes: Vec<String>,
    /// Expertos permitidos en todas las capas (modo experto, p. ej. OLMoE).
    #[serde(default)]
    pub experts: Vec<u64>,
    /// Expertos medidos por capa; si está presente, tiene prioridad sobre
    /// `experts`.
    #[serde(default)]
    pub experts_by_layer: Vec<Vec<u64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldTaxonomy {
    #[serde(default = "default_version")]
    pub version: u32,
    #[serde(default = "default_dim")]
    pub feature_dim: usize,
    pub worlds: Vec<WorldSpec>,
}

fn default_version() -> u32 {
    1
}

fn default_dim() -> usize {
    512
}

impl WorldTaxonomy {
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let data = std::fs::read(path)
            .map_err(|e| Error::new(format!("no se puede leer {}: {e}", path.display())))?;
        let tax: Self = serde_json::from_slice(&data)
            .map_err(|e| Error::new(format!("{}: JSON inválido: {e}", path.display())))?;
        tax.validate()?;
        Ok(tax)
    }

    pub fn validate(&self) -> Result<()> {
        if self.worlds.len() < 2 {
            return Err(Error::new("la taxonomía necesita al menos dos mundos"));
        }
        if self.feature_dim < 32 || self.feature_dim > 65536 {
            return Err(Error::new("feature_dim fuera de rango (32..65536)"));
        }
        let mut ids = std::collections::HashSet::new();
        for w in &self.worlds {
            if w.id.is_empty() || !ids.insert(w.id.as_str()) {
                return Err(Error::new(format!(
                    "mundo inválido o duplicado: {:?}",
                    w.id
                )));
            }
            if w.examples.is_empty() {
                return Err(Error::new(format!("el mundo '{}' no tiene ejemplos", w.id)));
            }
            for e in w.experts.iter().chain(w.experts_by_layer.iter().flatten()) {
                if *e >= 64 {
                    return Err(Error::new(format!(
                        "el mundo '{}' referencia el experto {e}; el máximo es 63",
                        w.id
                    )));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivatedWorld {
    pub id: String,
    pub label: String,
    pub priority: i32,
    pub provisional: bool,
    pub probability: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct Activation {
    pub worlds: Vec<ActivatedWorld>,
    pub incomplete: bool,
    pub warning: Option<String>,
    /// Probabilidades de todos los mundos, ordenadas de mayor a menor.
    pub probabilities: Vec<(String, f32)>,
}

#[derive(Debug, Clone)]
pub struct ActivationPolicy {
    pub threshold: f32,
    pub min_confidence: f32,
    /// Similitud coseno mínima con el centroide de algún mundo para aceptarlo.
    pub min_similarity: f32,
    pub max_worlds: usize,
    pub provisional_threshold: f32,
}

impl Default for ActivationPolicy {
    fn default() -> Self {
        Self {
            threshold: 0.30,
            min_confidence: 0.18,
            min_similarity: 0.14,
            max_worlds: 2,
            provisional_threshold: 0.08,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TinyRouter {
    pub feature_dim: usize,
    pub world_ids: Vec<String>,
    /// `worlds * feature_dim`, por filas.
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    /// Centroides normalizados para la puerta de novedad (`worlds * feature_dim`).
    #[serde(default)]
    pub centroids: Vec<f32>,
    /// Centroide global de todo el corpus de entrenamiento.
    #[serde(default)]
    pub global_centroid: Vec<f32>,
}

fn hash64(s: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn add_feature(v: &mut [f32], key: &str, weight: f32) {
    let h = hash64(key);
    let idx = (h as usize) % v.len();
    let sign = if h & (1u64 << 63) != 0 { -1.0 } else { 1.0 };
    v[idx] += sign * weight;
}

/// n-gramas de palabra, bigramas y trigramas de carácter con hashing firmado.
pub fn featurize(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .collect();
    for w in &words {
        add_feature(&mut v, &format!("w:{w}"), 1.0);
        // Prefijo: generaliza entre formas de la misma familia ("cocina",
        // "cocinar") y entre conjugaciones.
        let chars: Vec<char> = w.chars().collect();
        if chars.len() > 4 {
            let prefix: String = chars[..4].iter().collect();
            add_feature(&mut v, &format!("p:{prefix}"), 0.5);
        }
    }
    for pair in words.windows(2) {
        add_feature(&mut v, &format!("b:{} {}", pair[0], pair[1]), 0.8);
    }
    for w in &words {
        let s = format!("<{w}>");
        let chars: Vec<char> = s.chars().collect();
        if chars.len() >= 3 {
            for c in chars.windows(3) {
                add_feature(&mut v, &format!("c:{}", c.iter().collect::<String>()), 0.8);
            }
        }
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
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

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

impl TinyRouter {
    pub fn train(taxonomy: &WorldTaxonomy) -> Result<Self> {
        Self::train_with(taxonomy, 400, 0.5)
    }

    pub fn train_with(taxonomy: &WorldTaxonomy, epochs: usize, lr0: f32) -> Result<Self> {
        taxonomy.validate()?;
        let dim = taxonomy.feature_dim;
        let n = taxonomy.worlds.len();
        let mut samples: Vec<(Vec<f32>, usize)> = Vec::new();
        for (wi, world) in taxonomy.worlds.iter().enumerate() {
            for ex in &world.examples {
                samples.push((featurize(ex, dim), wi));
            }
        }
        if samples.is_empty() {
            return Err(Error::new("no hay ejemplos para entrenar el router"));
        }

        // Centroides por mundo: puerta de novedad para texto fuera de dominio.
        let mut centroids = vec![0.0f32; n * dim];
        let mut global_centroid = vec![0.0f32; dim];
        let mut counts = vec![0usize; n];
        for (x, y) in &samples {
            counts[*y] += 1;
            for d in 0..dim {
                centroids[y * dim + d] += x[d];
                global_centroid[d] += x[d];
            }
        }
        let total = samples.len().max(1) as f32;
        for v in global_centroid.iter_mut() {
            *v /= total;
        }
        normalize(&mut global_centroid);
        for k in 0..n {
            let c = counts[k].max(1) as f32;
            let slice = &mut centroids[k * dim..(k + 1) * dim];
            for v in slice.iter_mut() {
                *v /= c;
            }
            normalize(slice);
        }

        let mut rng = StdRng::seed_from_u64(0xCA8_217A);
        let mut weights = vec![0.0f32; n * dim];
        let mut bias = vec![0.0f32; n];
        let mut lr = lr0;
        let mut logits = vec![0.0f32; n];

        for _ in 0..epochs {
            samples.shuffle(&mut rng);
            for (x, y) in &samples {
                for k in 0..n {
                    logits[k] = bias[k] + dot(&weights[k * dim..(k + 1) * dim], x);
                }
                softmax_inplace(&mut logits);
                for k in 0..n {
                    let g = logits[k] - if k == *y { 1.0 } else { 0.0 };
                    if g.abs() < 1e-7 {
                        continue;
                    }
                    for d in 0..dim {
                        let w = &mut weights[k * dim + d];
                        *w -= lr * (g * x[d] + 1e-4 * *w);
                    }
                    bias[k] -= lr * g;
                }
            }
            lr *= 0.995;
        }

        Ok(Self {
            feature_dim: dim,
            world_ids: taxonomy.worlds.iter().map(|w| w.id.clone()).collect(),
            weights,
            bias,
            centroids,
            global_centroid,
        })
    }

    /// Similitud coseno del texto con el centroide global del corpus.
    pub fn global_similarity(&self, text: &str) -> f32 {
        let x = featurize(text, self.feature_dim);
        dot(&x, &self.global_centroid)
    }

    /// Similitud coseno del texto con el centroide de cada mundo.
    pub fn similarities(&self, text: &str) -> Vec<f32> {
        let x = featurize(text, self.feature_dim);
        let n = self.world_ids.len();
        (0..n)
            .map(|k| {
                dot(
                    &x,
                    &self.centroids[k * self.feature_dim..(k + 1) * self.feature_dim],
                )
            })
            .collect()
    }

    /// Puntuaciones lineales por mundo, sin softmax.
    pub fn logits(&self, text: &str) -> Vec<f32> {
        let x = featurize(text, self.feature_dim);
        (0..self.world_ids.len())
            .map(|k| {
                self.bias[k]
                    + dot(
                        &self.weights[k * self.feature_dim..(k + 1) * self.feature_dim],
                        &x,
                    )
            })
            .collect()
    }

    pub fn probabilities(&self, text: &str) -> Vec<f32> {
        let mut probs = self.logits(text);
        softmax_inplace(&mut probs);
        probs
    }

    pub fn activate(
        &self,
        taxonomy: &WorldTaxonomy,
        text: &str,
        policy: &ActivationPolicy,
    ) -> Activation {
        let logits = self.logits(text);
        let mut probs = logits.clone();
        softmax_inplace(&mut probs);
        let mut ranked: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        let probabilities: Vec<(String, f32)> = ranked
            .iter()
            .map(|&(i, p)| (taxonomy.worlds[i].id.clone(), p))
            .collect();

        let Some(&(top_idx, top_prob)) = ranked.first() else {
            return Activation {
                worlds: Vec::new(),
                incomplete: true,
                warning: Some("la taxonomía no tiene mundos".to_string()),
                probabilities,
            };
        };

        let sims = self.similarities(text);
        let best_sim = sims.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if top_prob < policy.min_confidence || best_sim < policy.min_similarity {
            return Activation {
                worlds: Vec::new(),
                incomplete: true,
                warning: Some(format!(
                    "mundo no encontrado (confianza {:.0}%, similitud {:.2}); modo incompleto",
                    top_prob * 100.0,
                    best_sim
                )),
                probabilities,
            };
        }

        let mut selected: Vec<usize> = ranked
            .iter()
            .filter(|(_, p)| *p >= policy.threshold)
            .take(policy.max_worlds)
            .map(|&(i, _)| i)
            .collect();
        let mut warning = None;
        if selected.is_empty() {
            selected.push(top_idx);
            warning = Some(format!(
                "confianza baja ({:.0}%); se activa solo '{}'",
                top_prob * 100.0,
                taxonomy.worlds[top_idx].id
            ));
        }

        // Mundos provisionales "por si acaso", sin desplazar a los titulares.
        for &(i, p) in &ranked {
            if p < policy.provisional_threshold || !taxonomy.worlds[i].provisional {
                continue;
            }
            if !selected.contains(&i) && selected.len() < policy.max_worlds + 2 {
                selected.push(i);
            }
        }

        selected.sort_by(|&a, &b| {
            taxonomy.worlds[b]
                .priority
                .cmp(&taxonomy.worlds[a].priority)
                .then(
                    probs[b]
                        .partial_cmp(&probs[a])
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        });

        let worlds = selected
            .iter()
            .map(|&i| {
                let w = &taxonomy.worlds[i];
                ActivatedWorld {
                    id: w.id.clone(),
                    label: w.label.clone().unwrap_or_else(|| w.id.clone()),
                    priority: w.priority,
                    provisional: w.provisional,
                    probability: probs[i],
                }
            })
            .collect();

        Activation {
            worlds,
            incomplete: false,
            warning,
            probabilities,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct EvalReport {
    pub train_examples: usize,
    pub test_examples: usize,
    pub correct: usize,
    pub accuracy: f32,
    pub per_world: BTreeMap<String, (usize, usize)>,
    pub errors: Vec<(String, String, String)>,
}

/// Entrena dejando fuera cada `holdout_every`-ésimo ejemplo de cada mundo y
/// mide la precisión top-1 en ese conjunto de prueba.
pub fn evaluate_holdout(
    taxonomy: &WorldTaxonomy,
    holdout_every: usize,
    policy: &ActivationPolicy,
) -> Result<EvalReport> {
    let mut train = WorldTaxonomy {
        version: taxonomy.version,
        feature_dim: taxonomy.feature_dim,
        worlds: Vec::new(),
    };
    let mut test: Vec<(String, String)> = Vec::new();
    for w in &taxonomy.worlds {
        let mut tw = w.clone();
        tw.examples.clear();
        for (i, ex) in w.examples.iter().enumerate() {
            if holdout_every > 0 && i % holdout_every == holdout_every - 1 {
                test.push((w.id.clone(), ex.clone()));
            } else {
                tw.examples.push(ex.clone());
            }
        }
        train.worlds.push(tw);
    }
    let router = TinyRouter::train(&train)?;
    let mut correct = 0usize;
    let mut per_world: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut errors = Vec::new();
    for (expected, text) in &test {
        let act = router.activate(&train, text, policy);
        let got = act
            .worlds
            .first()
            .map(|w| w.id.clone())
            .unwrap_or_else(|| "<incompleto>".to_string());
        let slot = per_world.entry(expected.clone()).or_insert((0, 0));
        slot.1 += 1;
        if &got == expected {
            correct += 1;
            slot.0 += 1;
        } else {
            errors.push((expected.clone(), got, text.clone()));
        }
    }
    let total = test.len();
    Ok(EvalReport {
        train_examples: train.worlds.iter().map(|w| w.examples.len()).sum(),
        test_examples: total,
        correct,
        accuracy: if total > 0 {
            correct as f32 / total as f32
        } else {
            0.0
        },
        per_world,
        errors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> WorldTaxonomy {
        WorldTaxonomy {
            version: 1,
            feature_dim: 256,
            worlds: vec![
                WorldSpec {
                    id: "codigo".into(),
                    label: Some("Código".into()),
                    priority: 10,
                    provisional: false,
                    examples: vec![
                        "escribe una función en Python".into(),
                        "tengo un bug en Rust".into(),
                        "revisa este código".into(),
                        "revisa el código de este proyecto".into(),
                        "cómo compilo el proyecto".into(),
                        "añade tests al repositorio".into(),
                        "error de sintaxis en JavaScript".into(),
                        "error de compilación en el proyecto".into(),
                        "refactoriza esta clase".into(),
                        "implementa una API en Go".into(),
                        "depura el programa".into(),
                        "depura este programa con errores".into(),
                    ],
                    routes: vec!["capa.0".into()],
                    experts: vec![],
                    experts_by_layer: vec![],
                },
                WorldSpec {
                    id: "matematicas".into(),
                    label: Some("Matemáticas".into()),
                    priority: 9,
                    provisional: false,
                    examples: vec![
                        "resuelve la integral".into(),
                        "calcula la derivada".into(),
                        "explica el teorema".into(),
                        "explica el teorema de Pitágoras".into(),
                        "halla la probabilidad".into(),
                        "resuelve esta ecuación".into(),
                        "calcula el área del círculo".into(),
                        "calcula el área del triángulo".into(),
                        "demuestra la conjetura".into(),
                        "factoriza el polinomio".into(),
                        "calcula el límite".into(),
                        "calcula el límite de la función".into(),
                    ],
                    routes: vec!["capa.1".into()],
                    experts: vec![],
                    experts_by_layer: vec![],
                },
                WorldSpec {
                    id: "cocina".into(),
                    label: Some("Cocina".into()),
                    priority: 4,
                    provisional: true,
                    examples: vec![
                        "receta de paella".into(),
                        "cómo hornear el pollo".into(),
                        "ingredientes de la tortilla".into(),
                        "ingredientes para la paella".into(),
                        "prepara una salsa".into(),
                        "postre de chocolate".into(),
                        "cocina arroz para cuatro".into(),
                        "cocina arroz para dos".into(),
                        "sofríe la cebolla".into(),
                        "hierve las patatas".into(),
                        "hornea el pan".into(),
                        "hornea el pan de centeno".into(),
                    ],
                    routes: vec!["capa.2".into()],
                    experts: vec![],
                    experts_by_layer: vec![],
                },
            ],
        }
    }

    #[test]
    fn trains_and_classifies_held_out() {
        let tax = fixture();
        let report = evaluate_holdout(&tax, 4, &ActivationPolicy::default()).unwrap();
        assert_eq!(report.train_examples, 27);
        assert_eq!(report.test_examples, 9);
        assert!(
            report.accuracy >= 0.66,
            "precisión baja: {} ({:?})",
            report.accuracy,
            report.errors
        );
        let router = TinyRouter::train(&tax).unwrap();
        let act = router.activate(
            &tax,
            "necesito arreglar un bug de Rust",
            &ActivationPolicy::default(),
        );
        assert!(!act.incomplete);
        assert_eq!(act.worlds[0].id, "codigo");
    }

    #[test]
    fn gibberish_falls_back_to_incomplete() {
        let tax = fixture();
        let router = TinyRouter::train(&tax).unwrap();
        let act = router.activate(&tax, "qzxv wplk ñandú 88", &ActivationPolicy::default());
        assert!(act.incomplete);
        assert!(act.warning.is_some());
        assert!(act.worlds.is_empty());
    }

    #[test]
    fn priority_orders_activated_worlds() {
        let tax = fixture();
        let router = TinyRouter::train(&tax).unwrap();
        let policy = ActivationPolicy {
            threshold: 0.05,
            max_worlds: 3,
            ..Default::default()
        };
        let act = router.activate(&tax, "integral en Python", &policy);
        assert!(!act.worlds.is_empty());
        let priorities: Vec<i32> = act.worlds.iter().map(|w| w.priority).collect();
        let mut sorted = priorities.clone();
        sorted.sort_by(|a, b| b.cmp(a));
        assert_eq!(priorities, sorted);
    }
}
