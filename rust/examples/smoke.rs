use std::path::Path;
use std::time::Instant;

use cabrita::model::{rss_bytes, KvCache, OlmoeModel};

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("uso: smoke <modelo.ukj> [texto]");
    let text = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "Hola".to_string());
    let t_open = Instant::now();
    let mut model = OlmoeModel::open(Path::new(&path)).expect("abrir modelo");
    println!(
        "abierto en {:.1} ms · densidad {} · {:.1} MB RSS",
        t_open.elapsed().as_secs_f64() * 1000.0,
        model.density(),
        rss_bytes() as f64 / 1048576.0
    );
    let cfg = model.config().clone();
    println!(
        "hidden={} layers={} heads={} kv={} hd={} inter={} experts={} top_k={} vocab={} ctx={}",
        cfg.hidden,
        cfg.layers,
        cfg.heads,
        cfg.kv_heads,
        cfg.head_dim,
        cfg.intermediate,
        cfg.experts,
        cfg.top_k,
        cfg.vocab,
        cfg.context
    );
    let tokens = model.tokenizer.build_prompt(
        None,
        &[cabrita::tokenizer::ChatMessage {
            role: cabrita::tokenizer::Role::User,
            content: text.clone(),
        }],
        true,
    );
    println!("tokens: {tokens:?}");
    println!("prompt: {:?}", model.tokenizer.decode(&tokens));
    let mut cache = KvCache::new(cfg.layers, cfg.kv_heads, cfg.head_dim, 32);
    let t0 = Instant::now();
    let logits = model.forward_tokens(&tokens, &mut cache).expect("forward");
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "forward {} tokens en {:.1} ms ({:.2} tok/s) · {:.1} MB RSS",
        tokens.len(),
        ms,
        tokens.len() as f64 / (ms / 1000.0),
        rss_bytes() as f64 / 1048576.0
    );
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    for &i in order.iter().take(12) {
        let s = model.tokenizer.token_text(i as u32).unwrap_or("<especial>");
        println!("{:6} {:>10.4}  {:?}", i, logits[i], s);
    }
}
