//! Mide qué expertos elige el router real de OLMoE para cada mundo.
//!
//! Uso: `experts <modelo.ukj> <worlds.json> <salida.json> [ejemplos_por_mundo]`
//! Escribe, por mundo, el top de expertos por capa para etiquetar las
//! particiones del `.ukj`.

use std::path::Path;

use cabrita::model::{KvCache, OlmoeModel};
use cabrita::router::WorldTaxonomy;
use cabrita::tokenizer::{ChatMessage, Role};
use serde_json::json;

fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().expect("modelo .ukj");
    let worlds_path = args.next().expect("taxonomía JSON");
    let out_path = args.next().expect("salida JSON");
    let per_world: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(3);

    let taxonomy = WorldTaxonomy::load(Path::new(&worlds_path)).expect("taxonomía");
    let mut model = OlmoeModel::open(Path::new(&model_path)).expect("modelo");
    let cfg = model.config().clone();

    let mut report = serde_json::Map::new();
    for world in &taxonomy.worlds {
        let mut per_layer = vec![[0u64; 64]; cfg.layers];
        let mut used = 0usize;
        for example in world.examples.iter().take(per_world) {
            let tokens = model.tokenizer.build_prompt(
                None,
                &[ChatMessage {
                    role: Role::User,
                    content: example.clone(),
                }],
                true,
            );
            let mut cache = KvCache::new(cfg.layers, cfg.kv_heads, cfg.head_dim, 64);
            model.start_usage();
            model.forward_tokens(&tokens, &mut cache).expect("forward");
            let usage = model.take_usage().expect("uso");
            for (layer, counts) in usage.iter().enumerate() {
                for (e, &c) in counts.iter().enumerate() {
                    per_layer[layer][e] += c;
                }
            }
            used += 1;
            eprintln!("{}: {used} ejemplos", world.id);
        }

        let mut layers_json = Vec::new();
        for counts in &per_layer {
            let mut ranked: Vec<(usize, u64)> = counts.iter().copied().enumerate().collect();
            ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            let top: Vec<usize> = ranked
                .iter()
                .filter(|&&(_, c)| c > 0)
                .take(16)
                .map(|&(e, _)| e)
                .collect();
            layers_json.push(top);
        }
        report.insert(world.id.clone(), json!(layers_json));
    }

    let data = serde_json::to_string_pretty(&json!({
        "model": model_path,
        "examples_per_world": per_world,
        "experts_by_layer": report,
    }))
    .expect("json");
    std::fs::write(&out_path, data).expect("escribir salida");
    println!("escrito {out_path}");
}
