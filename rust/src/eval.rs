//! Evaluación reproducible: perplejidad y QA corto de opción múltiple.
//!
//! La perplejidad se calcula token a token con la misma ruta de inferencia que
//! `chat`. El QA compara la log-verosimilitud media de cada opción.

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::model::{KvCache, OlmoeModel};

#[derive(Debug, Clone, Serialize)]
pub struct PerplexityReport {
    pub model: String,
    pub density: String,
    pub tokens: usize,
    pub nll: f64,
    pub perplexity: f64,
    pub text_bytes: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QaItem {
    #[serde(default)]
    pub id: Option<String>,
    pub prompt: String,
    pub choices: Vec<String>,
    pub answer: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct QaItemResult {
    pub id: String,
    pub expected: String,
    pub predicted: String,
    pub correct: bool,
    pub scores: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct QaReport {
    pub model: String,
    pub density: String,
    pub total: usize,
    pub correct: usize,
    pub accuracy: f32,
    pub items: Vec<QaItemResult>,
}

fn log_softmax_at(logits: &[f32], target: usize) -> Result<f64> {
    if target >= logits.len() {
        return Err(Error::new(format!(
            "token objetivo {target} fuera del vocabulario ({})",
            logits.len()
        )));
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|&v| (v as f64 - max).exp()).sum();
    Ok((logits[target] as f64 - max) - sum.ln())
}

fn new_cache(model: &OlmoeModel, ctx: usize) -> KvCache {
    let cfg = model.config();
    KvCache::new(cfg.layers, cfg.kv_heads, cfg.head_dim, ctx)
}

/// Perplejidad sobre un texto; el texto debe caber en `ctx` tokens.
pub fn perplexity(model: &mut OlmoeModel, text: &str, ctx: usize) -> Result<PerplexityReport> {
    let tokens = model.tokenizer.encode(text);
    if tokens.len() < 2 {
        return Err(Error::new(
            "el texto de evaluación necesita al menos dos tokens",
        ));
    }
    if tokens.len() > ctx {
        return Err(Error::new(format!(
            "el texto tiene {} tokens y no cabe en ctx={ctx}",
            tokens.len()
        )));
    }
    let mut cache = new_cache(model, ctx);
    let mut nll = 0.0f64;
    for i in 0..tokens.len() - 1 {
        let logits = model.forward_token(tokens[i], &mut cache)?;
        nll -= log_softmax_at(&logits, tokens[i + 1] as usize)?;
    }
    let predicted = (tokens.len() - 1) as f64;
    Ok(PerplexityReport {
        model: model.ukj.model_id().to_string(),
        density: model.density().to_string(),
        tokens: tokens.len() - 1,
        nll,
        perplexity: (nll / predicted).exp(),
        text_bytes: text.len(),
    })
}

fn continuation_score(
    model: &mut OlmoeModel,
    prompt: &str,
    choice: &str,
    ctx: usize,
) -> Result<f32> {
    let prompt_tokens = model.tokenizer.encode(prompt);
    let full = model.tokenizer.encode(&format!("{prompt} {choice}"));
    if full.len() <= prompt_tokens.len() || !full.starts_with(&prompt_tokens) {
        return Err(Error::new(format!(
            "la opción '{choice}' no continúa el prompt de forma tokenizable"
        )));
    }
    if full.len() > ctx {
        return Err(Error::new(format!(
            "la opción '{choice}' no cabe en ctx={ctx} ({} tokens)",
            full.len()
        )));
    }
    let mut cache = new_cache(model, ctx);
    let mut nll = 0.0f64;
    let mut count = 0usize;
    for i in 0..full.len() - 1 {
        let logits = model.forward_token(full[i], &mut cache)?;
        if i + 1 >= prompt_tokens.len() {
            nll -= log_softmax_at(&logits, full[i + 1] as usize)?;
            count += 1;
        }
    }
    if count == 0 {
        return Err(Error::new("la opción no aporta tokens evaluables"));
    }
    Ok((-(nll / count as f64)) as f32)
}

/// QA de opción múltiple por log-verosimilitud media de cada opción.
pub fn qa(model: &mut OlmoeModel, items: &[QaItem], ctx: usize) -> Result<QaReport> {
    if items.is_empty() {
        return Err(Error::new("el archivo de QA no contiene preguntas"));
    }
    let mut results = Vec::with_capacity(items.len());
    let mut correct = 0usize;
    for item in items {
        if item.choices.len() < 2 {
            return Err(Error::new(format!(
                "la pregunta {:?} necesita al menos dos opciones",
                item.id
            )));
        }
        let mut scores = Vec::with_capacity(item.choices.len());
        for choice in &item.choices {
            scores.push(continuation_score(model, &item.prompt, choice, ctx)?);
        }
        let best = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let predicted = item.choices[best].clone();
        let is_correct = predicted == item.answer;
        if is_correct {
            correct += 1;
        }
        results.push(QaItemResult {
            id: item
                .id
                .clone()
                .unwrap_or_else(|| format!("q{}", results.len() + 1)),
            expected: item.answer.clone(),
            predicted,
            correct: is_correct,
            scores,
        });
    }
    Ok(QaReport {
        model: model.ukj.model_id().to_string(),
        density: model.density().to_string(),
        total: items.len(),
        correct,
        accuracy: correct as f32 / items.len() as f32,
        items: results,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_softmax_is_negative_and_normalized() {
        let logits = [0.0f32, 1.0, 2.0];
        let mut total = 0.0f64;
        for i in 0..3 {
            let v = log_softmax_at(&logits, i).unwrap();
            assert!(v <= 0.0);
            total += v.exp();
        }
        assert!((total - 1.0).abs() < 1e-6);
    }
}
