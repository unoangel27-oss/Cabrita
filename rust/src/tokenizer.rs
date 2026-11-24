//! Tokenizer GPT-2/OLMo (BPE byte-level) leído de la cabecera UKJ.

use std::collections::HashMap;

use fancy_regex::Regex;
use serde_json::Value;

use crate::error::{Error, Result};
use crate::format::Header;

const PRETOKEN: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    ranks: HashMap<(String, String), usize>,
    id_to_token: Vec<String>,
    token_type: Vec<u8>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    bos: u32,
    eos: u32,
    pub vocab_size: usize,
    pre: Regex,
}

fn bytes_to_unicode() -> ([char; 256], HashMap<char, u8>) {
    let mut bs: Vec<u16> = (b'!' as u16..=b'~' as u16).collect();
    bs.extend(b'\xA1' as u16..=b'\xAC' as u16);
    bs.extend(b'\xAE' as u16..=b'\xFF' as u16);
    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n = 0u32;
    for b in 0u16..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut map = ['\0'; 256];
    let mut inverse = HashMap::new();
    for (b, c) in bs.iter().zip(cs.iter()) {
        let ch = char::from_u32(*c).expect("code point válido");
        map[*b as usize] = ch;
        inverse.insert(ch, *b as u8);
    }
    (map, inverse)
}

impl Tokenizer {
    pub fn from_header(header: &Header) -> Result<Self> {
        let meta = header
            .model
            .metadata
            .as_object()
            .ok_or_else(|| Error::new("la cabecera no contiene model.metadata"))?;
        let tokens = meta
            .get("tokenizer.ggml.tokens")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::new("falta tokenizer.ggml.tokens"))?;
        let merges = meta
            .get("tokenizer.ggml.merges")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::new("falta tokenizer.ggml.merges"))?;
        let token_type = meta
            .get("tokenizer.ggml.token_type")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .map(|v| v.as_u64().unwrap_or(1) as u8)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let mut vocab = HashMap::with_capacity(tokens.len());
        let mut id_to_token = Vec::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            let s = t
                .as_str()
                .ok_or_else(|| Error::new(format!("token {i} no es string")))?;
            vocab.insert(s.to_string(), i as u32);
            id_to_token.push(s.to_string());
        }
        let mut ranks = HashMap::with_capacity(merges.len());
        for (i, m) in merges.iter().enumerate() {
            let s = m
                .as_str()
                .ok_or_else(|| Error::new(format!("merge {i} no es string")))?;
            if let Some((a, b)) = s.split_once(' ') {
                ranks.insert((a.to_string(), b.to_string()), i);
            }
        }
        let bos = meta
            .get("tokenizer.ggml.bos_token_id")
            .and_then(Value::as_u64)
            .unwrap_or(50279) as u32;
        let eos = meta
            .get("tokenizer.ggml.eos_token_id")
            .and_then(Value::as_u64)
            .unwrap_or(bos as u64) as u32;
        let (byte_to_char, char_to_byte) = bytes_to_unicode();
        let pre = Regex::new(PRETOKEN)
            .map_err(|e| Error::new(format!("regex de pre-tokenización inválida: {e}")))?;
        Ok(Self {
            vocab,
            ranks,
            id_to_token,
            token_type,
            byte_to_char,
            char_to_byte,
            bos,
            eos,
            vocab_size: tokens.len(),
            pre,
        })
    }

    pub fn bos(&self) -> u32 {
        self.bos
    }

    pub fn eos(&self) -> u32 {
        self.eos
    }

    fn encode_chunk(&self, chunk: &str, out: &mut Vec<u32>) {
        let mut symbols: Vec<String> = chunk
            .as_bytes()
            .iter()
            .map(|b| self.byte_to_char[*b as usize].to_string())
            .collect();
        while symbols.len() > 1 {
            let mut best: Option<(usize, usize)> = None;
            for i in 0..symbols.len() - 1 {
                if let Some(&rank) = self
                    .ranks
                    .get(&(symbols[i].clone(), symbols[i + 1].clone()))
                {
                    if best.map_or(true, |(r, _)| rank < r) {
                        best = Some((rank, i));
                    }
                }
            }
            let Some((_, at)) = best else { break };
            let merged = format!("{}{}", symbols[at], symbols[at + 1]);
            symbols.splice(at..at + 2, [merged]);
        }
        for s in symbols {
            if let Some(&id) = self.vocab.get(&s) {
                out.push(id);
            } else {
                // Fallback: byte a byte; el vocabulario OLMo contiene los 256 bytes.
                for ch in s.chars() {
                    if let Some(&b) = self.char_to_byte.get(&ch) {
                        let key = self.byte_to_char[b as usize].to_string();
                        if let Some(&id) = self.vocab.get(&key) {
                            out.push(id);
                        }
                    }
                }
            }
        }
    }

    fn encode_pretokenized(&self, text: &str, out: &mut Vec<u32>) {
        let mut cursor = 0usize;
        for m in self.pre.find_iter(text).flatten() {
            if m.start() > cursor {
                self.encode_chunk(&text[cursor..m.start()], out);
            }
            self.encode_chunk(m.as_str(), out);
            cursor = m.end();
        }
        if cursor < text.len() {
            self.encode_chunk(&text[cursor..], out);
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Los runs de 2..24 espacios son tokens añadidos y se consumen antes
        // de la pre-tokenización; el resto sigue el regex ByteLevel de OLMo.
        let mut out = Vec::new();
        let bytes = text.as_bytes();
        let mut start = 0usize;
        let mut i = 0usize;
        while i < bytes.len() {
            if bytes[i] != b' ' {
                i += 1;
                continue;
            }
            let run_start = i;
            while i < bytes.len() && bytes[i] == b' ' {
                i += 1;
            }
            let run = i - run_start;
            if run < 2 {
                continue;
            }
            if run_start > start {
                self.encode_pretokenized(&text[start..run_start], &mut out);
            }
            let mut rem = run;
            while rem >= 2 {
                let take = rem.min(24);
                out.push(50278 - take as u32);
                rem -= take;
            }
            // Si sobra un espacio, se deja con el texto siguiente para que
            // la pre-tokenización lo una a la palabra (" y" -> "Ġy").
            start = if rem == 1 { i - 1 } else { i };
        }
        if start < text.len() {
            self.encode_pretokenized(&text[start..], &mut out);
        }
        out
    }

    /// Texto de un token, o `None` si es de control/desconocido.
    pub fn token_text(&self, id: u32) -> Option<&str> {
        let s = self.id_to_token.get(id as usize)?;
        let ty = self.token_type.get(id as usize).copied().unwrap_or(1);
        if ty == 3 || ty == 5 {
            return None;
        }
        Some(s)
    }

    pub fn decode_bytes(&self, id: u32) -> Vec<u8> {
        let Some(s) = self.token_text(id) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for ch in s.chars() {
            if let Some(&b) = self.char_to_byte.get(&ch) {
                out.push(b);
            } else {
                out.extend_from_slice(ch.to_string().as_bytes());
            }
        }
        out
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(&self.decode_bytes(id));
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn build_prompt(
        &self,
        system: Option<&str>,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Vec<u32> {
        let mut out = vec![self.bos];
        let mut all: Vec<ChatMessage> = Vec::with_capacity(messages.len() + 1);
        if let Some(system) = system {
            if !system.trim().is_empty() && messages.first().map(|m| m.role) != Some(Role::System) {
                all.push(ChatMessage {
                    role: Role::System,
                    content: system.to_string(),
                });
            }
        }
        all.extend_from_slice(messages);
        for m in &all {
            let (tag, add_eos) = match m.role {
                Role::System => ("<|system|>\n", false),
                Role::User => ("<|user|>\n", false),
                Role::Assistant => ("<|assistant|>\n", true),
            };
            out.extend(self.encode(tag));
            out.extend(self.encode(&m.content));
            if add_eos {
                out.push(self.eos);
            }
        }
        if add_generation_prompt {
            out.extend(self.encode("<|assistant|>"));
        }
        out
    }
}

/// Acumula bytes de tokens y emite solo UTF-8 completo. No retiene el
/// tokenizer, así que puede vivir dentro de una sesión que también muta el
/// modelo.
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    pub fn push_bytes(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    break;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    if valid > 0 {
                        out.push_str(
                            std::str::from_utf8(&self.pending[..valid])
                                .expect("prefijo UTF-8 validado"),
                        );
                        self.pending.drain(..valid);
                    }
                    if e.error_len().is_some() {
                        out.push('\u{fffd}');
                        self.pending.drain(..1);
                    } else {
                        break;
                    }
                }
            }
        }
        out
    }

    pub fn finish(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}

pub struct StreamDecoder<'a> {
    tok: &'a Tokenizer,
    pending: Vec<u8>,
}

impl<'a> StreamDecoder<'a> {
    pub fn new(tok: &'a Tokenizer) -> Self {
        Self {
            tok,
            pending: Vec::new(),
        }
    }

    pub fn push(&mut self, id: u32) -> String {
        self.pending.extend(self.tok.decode_bytes(id));
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    break;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    if valid > 0 {
                        out.push_str(
                            std::str::from_utf8(&self.pending[..valid])
                                .expect("prefijo UTF-8 validado"),
                        );
                        self.pending.drain(..valid);
                    }
                    if e.error_len().is_some() {
                        // Byte inválido aislado: se reemplaza y se sigue.
                        out.push('\u{fffd}');
                        self.pending.drain(..1);
                    } else {
                        break;
                    }
                }
            }
        }
        out
    }

    pub fn finish(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_unicode_roundtrip() {
        let (m, inv) = bytes_to_unicode();
        for b in 0..=255u8 {
            assert_eq!(inv.get(&m[b as usize]), Some(&b));
        }
    }
}
