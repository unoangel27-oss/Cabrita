use std::fs::File;
use std::io::Write;
use std::path::Path;

use cabrita::model::{KvCache, OlmoeModel};
use cabrita::tokenizer::{ChatMessage, Role};

fn main() {
    let path = std::env::args().nth(1).expect("modelo");
    let out = std::env::args().nth(2).expect("salida");
    let text = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "Hola".to_string());
    let mut model = OlmoeModel::open(Path::new(&path)).expect("abrir");
    let tokens = model.tokenizer.build_prompt(
        None,
        &[ChatMessage {
            role: Role::User,
            content: text,
        }],
        true,
    );
    let cfg = model.config().clone();
    let mut cache = KvCache::new(cfg.layers, cfg.kv_heads, cfg.head_dim, 64);
    let logits = model.forward_tokens(&tokens, &mut cache).expect("forward");
    let mut f = File::create(&out).expect("crear salida");
    f.write_all(&(tokens.len() as u32).to_le_bytes()).unwrap();
    for t in &tokens {
        f.write_all(&t.to_le_bytes()).unwrap();
    }
    for l in &logits {
        f.write_all(&l.to_le_bytes()).unwrap();
    }
    println!("tokens={tokens:?}");
    println!("logits={} -> {out}", logits.len());
}
