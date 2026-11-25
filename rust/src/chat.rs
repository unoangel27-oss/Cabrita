//! Sesión de chat, muestreo y métricas de inferencia CPU.

use std::collections::VecDeque;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;

use crate::error::{Error, Result};
use crate::model::{peak_rss_bytes, rss_bytes, KvCache, OlmoeModel};
use crate::tokenizer::{ChatMessage, Role, Utf8Stream};

#[derive(Debug, Clone)]
pub struct GenOptions {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
    pub stream: bool,
}

impl Default for GenOptions {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.1,
            seed: 42,
            stream: true,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TurnMetrics {
    pub density: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub ttft_ms: f64,
    pub latency_ms: f64,
    pub tokens_per_s: f64,
    pub total_tokens_per_s: f64,
    pub rss_mb: f64,
    pub peak_rss_mb: f64,
    pub kv_cache_mb: f64,
    pub ctx_used: usize,
    pub ctx_capacity: usize,
    pub stop: String,
}

pub struct Sampler {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    rng: StdRng,
    recent: VecDeque<u32>,
}

impl Sampler {
    pub fn new(opts: &GenOptions) -> Self {
        Self {
            temperature: opts.temperature.max(0.0),
            top_k: opts.top_k,
            top_p: opts.top_p.clamp(0.0, 1.0),
            repetition_penalty: opts.repetition_penalty.max(1.0),
            rng: StdRng::seed_from_u64(opts.seed),
            recent: VecDeque::with_capacity(64),
        }
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        let n = logits.len();
        if n == 0 {
            return 0;
        }
        let mut scores = logits.to_vec();

        if self.repetition_penalty > 1.0 {
            for &id in &self.recent {
                let i = id as usize;
                if i >= n {
                    continue;
                }
                if scores[i] > 0.0 {
                    scores[i] /= self.repetition_penalty;
                } else {
                    scores[i] *= self.repetition_penalty;
                }
            }
        }

        let token = if self.temperature <= 0.0 {
            argmax(&scores) as u32
        } else {
            for s in scores.iter_mut() {
                *s /= self.temperature;
            }
            softmax(&mut scores);
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| {
                scores[b]
                    .partial_cmp(&scores[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });

            let keep = if self.top_k > 0 && self.top_k < n {
                self.top_k
            } else {
                n
            };
            order.truncate(keep);

            if self.top_p < 1.0 {
                let mut cum = 0.0f32;
                let mut cut = order.len();
                for (i, &idx) in order.iter().enumerate() {
                    cum += scores[idx];
                    if cum >= self.top_p {
                        cut = i + 1;
                        break;
                    }
                }
                order.truncate(cut.max(1));
            }

            let total: f32 = order.iter().map(|&i| scores[i]).sum();
            if total <= 0.0 {
                order[0] as u32
            } else {
                let mut r = self.rng.random::<f32>() * total;
                let mut chosen = order[0];
                for &idx in &order {
                    chosen = idx;
                    r -= scores[idx];
                    if r <= 0.0 {
                        break;
                    }
                }
                chosen as u32
            }
        };

        if self.recent.len() == 64 {
            self.recent.pop_front();
        }
        self.recent.push_back(token);
        token
    }
}

fn argmax(xs: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &v) in xs.iter().enumerate() {
        if v > xs[best] {
            best = i;
        }
    }
    best
}

fn softmax(xs: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &v in xs.iter() {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for v in xs.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in xs.iter_mut() {
            *v /= sum;
        }
    }
}

pub struct ChatSession {
    pub model: OlmoeModel,
    pub cache: KvCache,
    pub tokens: Vec<u32>,
    pub messages: Vec<ChatMessage>,
    pub system: Option<String>,
}

impl ChatSession {
    pub fn new(model: OlmoeModel, ctx: usize, system: Option<String>) -> Result<Self> {
        let cfg = model.config();
        let capacity = ctx.min(cfg.context).max(8);
        let cache = KvCache::new(cfg.layers, cfg.kv_heads, cfg.head_dim, capacity);
        Ok(Self {
            model,
            cache,
            tokens: Vec::new(),
            messages: Vec::new(),
            system,
        })
    }

    pub fn reset(&mut self) {
        self.cache.reset();
        self.tokens.clear();
        self.messages.clear();
    }

    pub fn set_system(&mut self, system: Option<String>) {
        self.system = system;
        self.cache.reset();
        self.tokens.clear();
    }

    pub fn respond<F: FnMut(&str)>(
        &mut self,
        user: &str,
        opts: &GenOptions,
        on_token: F,
    ) -> Result<(String, TurnMetrics)> {
        self.messages.push(ChatMessage {
            role: Role::User,
            content: user.to_string(),
        });
        let prompt =
            self.model
                .tokenizer
                .build_prompt(self.system.as_deref(), &self.messages, true);

        self.generate(prompt, opts, on_token)
    }

    /// Responde a una lista completa de mensajes (API compatible OpenAI).
    pub fn respond_chat<F: FnMut(&str)>(
        &mut self,
        messages: &[ChatMessage],
        opts: &GenOptions,
        on_token: F,
    ) -> Result<(String, TurnMetrics)> {
        self.cache.reset();
        self.tokens.clear();
        self.messages = messages.to_vec();
        let prompt = self.model.tokenizer.build_prompt(None, messages, true);
        self.generate(prompt, opts, on_token)
    }

    fn generate<F: FnMut(&str)>(
        &mut self,
        prompt: Vec<u32>,
        opts: &GenOptions,
        mut on_token: F,
    ) -> Result<(String, TurnMetrics)> {
        if prompt.len() >= self.cache.capacity() {
            return Err(Error::new(format!(
                "el prompt ocupa {} tokens y el contexto es {}; usa --ctx o acorta la conversación",
                prompt.len(),
                self.cache.capacity()
            )));
        }

        let common = self
            .tokens
            .iter()
            .zip(&prompt)
            .take_while(|(a, b)| a == b)
            .count();
        if common < self.tokens.len() {
            self.cache.reset();
            self.tokens.clear();
        }
        let suffix = &prompt[self.tokens.len()..];

        let start = Instant::now();
        let mut logits = self.model.forward_tokens(suffix, &mut self.cache)?;
        self.tokens.extend_from_slice(suffix);
        let prefill_ms = start.elapsed().as_secs_f64() * 1000.0;

        let mut sampler = Sampler::new(opts);
        let mut stream = Utf8Stream::new();
        let mut response = String::new();
        let mut generated = 0usize;
        let mut decode_s = 0.0f64;
        let mut ttft_ms = 0.0f64;
        let budget = opts
            .max_tokens
            .min(self.cache.capacity().saturating_sub(prompt.len()));
        let mut stop = "max_tokens".to_string();

        for i in 0..budget {
            let token = sampler.sample(&logits);
            if i == 0 {
                ttft_ms = start.elapsed().as_secs_f64() * 1000.0;
            }
            self.tokens.push(token);
            generated += 1;

            if token == self.model.tokenizer.eos() {
                let t = Instant::now();
                self.model.forward_token(token, &mut self.cache)?;
                decode_s += t.elapsed().as_secs_f64();
                stop = "eos".to_string();
                break;
            }

            let piece = {
                let bytes = self.model.tokenizer.decode_bytes(token);
                stream.push_bytes(&bytes)
            };
            if !piece.is_empty() {
                response.push_str(&piece);
                if opts.stream {
                    on_token(&piece);
                }
            }

            let t = Instant::now();
            logits = self.model.forward_token(token, &mut self.cache)?;
            decode_s += t.elapsed().as_secs_f64();
            if i + 1 == budget {
                stop = if budget < opts.max_tokens {
                    "context".to_string()
                } else {
                    "max_tokens".to_string()
                };
            }
        }

        let tail = stream.finish();
        if !tail.is_empty() {
            response.push_str(&tail);
            if opts.stream {
                on_token(&tail);
            }
        }

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        let tokens_per_s = if decode_s > 0.0 {
            generated as f64 / decode_s
        } else {
            0.0
        };
        let total_tokens_per_s = if latency_ms > 0.0 {
            generated as f64 / (latency_ms / 1000.0)
        } else {
            0.0
        };

        self.messages.push(ChatMessage {
            role: Role::Assistant,
            content: response.clone(),
        });

        let metrics = TurnMetrics {
            density: self.model.density().to_string(),
            prompt_tokens: prompt.len(),
            generated_tokens: generated,
            prefill_ms,
            decode_ms: decode_s * 1000.0,
            ttft_ms,
            latency_ms,
            tokens_per_s,
            total_tokens_per_s,
            rss_mb: rss_bytes() as f64 / (1024.0 * 1024.0),
            peak_rss_mb: peak_rss_bytes() as f64 / (1024.0 * 1024.0),
            kv_cache_mb: self.cache.bytes() as f64 / (1024.0 * 1024.0),
            ctx_used: self.tokens.len(),
            ctx_capacity: self.cache.capacity(),
            stop,
        };
        Ok((response, metrics))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampler_greedy_picks_argmax() {
        let opts = GenOptions {
            temperature: 0.0,
            ..Default::default()
        };
        let mut s = Sampler::new(&opts);
        let logits = [0.1f32, 3.0, -1.0, 2.0];
        assert_eq!(s.sample(&logits), 1);
        assert_eq!(s.sample(&logits), 1);
    }

    #[test]
    fn softmax_is_stable_with_large_logits() {
        let mut v = [1000.0f32, 1000.0, 999.0];
        softmax(&mut v);
        assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(v[0] > v[2]);
    }
}
