use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand};

use cabrita::chat::{ChatSession, GenOptions, TurnMetrics};
use cabrita::model::{rss_bytes, OlmoeModel};
use cabrita::operators::{read_log, OperatorConfig, OperatorEvent, SessionLog, TicsTracker};
use cabrita::router::{evaluate_holdout, Activation, ActivationPolicy, TinyRouter, WorldTaxonomy};
use cabrita::routes::{build_plan, LoadStats};
use cabrita::{
    ensure_subs, extract_subs, format_summary, inspect_report, verify_report, verify_subs, UkjFile,
};

#[derive(Parser)]
#[command(
    name = "cabrita",
    version,
    about = "Runtime y herramientas del formato UKJ"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Rutas, tensores, operadores, densidades y checksums de un .ukj
    Inspect(InspectArgs),
    /// Valida el contenedor de principio a fin
    Verify(VerifyArgs),
    /// Extrae, verifica o repara el .subs de un modelo
    Subs(SubsArgs),
    /// Chat CPU nativo con streaming y métricas
    Chat(ChatArgs),
    /// Genera tokens y reporta métricas sin REPL
    Bench(BenchArgs),
    /// Activa mundos, construye el plan de rutas y mide la carga perezosa
    Route(RouteArgs),
    /// Operadores de Fase 0: tics, stop-subparameter y continue parameter
    Tics(TicsArgs),
    /// Diagnóstico del backend Vulkan
    Vulkan(VulkanArgs),
}

#[derive(Args)]
struct InspectArgs {
    file: PathBuf,
    #[arg(long)]
    json: bool,
    #[arg(long)]
    tensors: bool,
    #[arg(long)]
    deep: bool,
    #[arg(long)]
    subs: bool,
    #[arg(long)]
    subs_dir: Option<PathBuf>,
    /// Muestra el log de operadores de la última sesión
    #[arg(long)]
    log: bool,
    /// Log de operadores concreto (implica --log)
    #[arg(long)]
    log_file: Option<PathBuf>,
}

#[derive(Args)]
struct VerifyArgs {
    file: PathBuf,
    #[arg(long)]
    quick: bool,
    #[arg(long)]
    subs: bool,
    #[arg(long)]
    subs_dir: Option<PathBuf>,
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct SubsArgs {
    #[command(subcommand)]
    command: SubsCommand,
}

#[derive(Args)]
struct ChatArgs {
    /// Modelo .ukj
    file: PathBuf,
    /// Una sola petición; sin este flag arranca el REPL
    #[arg(long)]
    prompt: Option<String>,
    /// System prompt inyectable
    #[arg(long)]
    system: Option<String>,
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,
    #[arg(long, default_value_t = 40)]
    top_k: usize,
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,
    #[arg(long, default_value_t = 1.1)]
    repetition_penalty: f32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 1024)]
    ctx: usize,
    /// Salida JSON (respuesta + métricas)
    #[arg(long)]
    json: bool,
    #[arg(long)]
    no_stream: bool,
    #[arg(long)]
    no_metrics: bool,
    /// Taxonomía de mundos para restringir los expertos del MoE. Si se omite,
    /// se busca worlds/olmoe.json en el directorio actual o sus padres.
    #[arg(long)]
    worlds: Option<PathBuf>,
    /// Mundo(s) cuyos expertos se permiten (repetible)
    #[arg(long = "world")]
    world: Vec<String>,
    /// Diagnóstico: permite los 64 expertos con la ruta enmascarada
    #[arg(long)]
    mask_all: bool,
    /// Desactiva la renormalización de pesos MoE con máscara
    #[arg(long)]
    no_renorm: bool,
    /// Omite la verificación SHA-256 del payload y del trailer al abrir
    #[arg(long)]
    no_verify_payload: bool,
    /// Activa los tics y el operador stop-subparameter
    #[arg(long)]
    tics: bool,
    /// Nivel normal de tics (activaciones/s)
    #[arg(long, default_value_t = 2.0)]
    tics_normal: f32,
    /// Nivel anómalo de tics (activaciones/s)
    #[arg(long, default_value_t = 20.0)]
    tics_anomalous: f32,
    /// Escribe el log de operadores en ~/.cabrita/logs
    #[arg(long)]
    log: bool,
    /// Ruta concreta del log de operadores (implica --log)
    #[arg(long)]
    log_file: Option<PathBuf>,
    /// Backend de inferencia
    #[arg(long, default_value = "cpu", value_parser = ["cpu", "vulkan"])]
    backend: String,
    /// Presupuesto de caché de pesos en Vulkan (MB)
    #[arg(long, default_value_t = 4096)]
    vulkan_cache_mb: u64,
}

#[derive(Args)]
struct BenchArgs {
    /// Modelo .ukj
    file: PathBuf,
    #[arg(
        long,
        default_value = "Explain what a transformer is in one paragraph."
    )]
    prompt: String,
    #[arg(long, default_value_t = 32)]
    tokens: usize,
    #[arg(long, default_value_t = 1024)]
    ctx: usize,
    #[arg(long)]
    json: bool,
    /// Backend de inferencia
    #[arg(long, default_value = "cpu", value_parser = ["cpu", "vulkan"])]
    backend: String,
    /// Presupuesto de caché de pesos en Vulkan (MB)
    #[arg(long, default_value_t = 4096)]
    vulkan_cache_mb: u64,
}

#[derive(Args)]
struct TicsArgs {
    /// Demostración sintética del ciclo normal → anómalo → offload
    #[arg(long)]
    demo: bool,
    /// Modelo .ukj para ejecutar una petición con tics reales
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long, default_value = "Escribe una función Rust que sume dos números.")]
    prompt: String,
    #[arg(long, default_value_t = 64)]
    max_tokens: usize,
    #[arg(long, default_value_t = 1024)]
    ctx: usize,
    #[arg(long, default_value_t = 2.0)]
    normal: f32,
    #[arg(long, default_value_t = 20.0)]
    anomalous: f32,
    /// Escribe el log de operadores en ~/.cabrita/logs
    #[arg(long)]
    log: bool,
    /// Ruta concreta del log de operadores (implica --log)
    #[arg(long)]
    log_file: Option<PathBuf>,
    #[arg(long)]
    json: bool,
    /// Backend de inferencia
    #[arg(long, default_value = "cpu", value_parser = ["cpu", "vulkan"])]
    backend: String,
    /// Presupuesto de caché de pesos en Vulkan (MB)
    #[arg(long, default_value_t = 4096)]
    vulkan_cache_mb: u64,
}

#[derive(Args)]
struct VulkanArgs {
    /// Muestra el adaptador Vulkan detectado
    #[arg(long)]
    probe: bool,
    /// Compara el shader con la ruta CPU en F32/F16/Q2_K/Q3_K/Q6_K
    #[arg(long)]
    selftest: bool,
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct RouteArgs {
    /// Modelo .ukj
    file: PathBuf,
    /// Taxonomía de mundos
    #[arg(long, default_value = "worlds/olmoe.json")]
    worlds: PathBuf,
    /// Consulta a clasificar
    #[arg(long)]
    text: Option<String>,
    /// Evalúa el router con ejemplos reservados
    #[arg(long)]
    eval: bool,
    /// Uno de cada N ejemplos de cada mundo queda para evaluación
    #[arg(long, default_value_t = 4)]
    holdout_every: usize,
    /// Carga las páginas del plan y mide el RSS
    #[arg(long)]
    load: bool,
    /// Libera el plan después de cargarlo
    #[arg(long)]
    unload: bool,
    /// Umbral de probabilidad para activar un mundo
    #[arg(long, default_value_t = 0.30)]
    threshold: f32,
    /// Máximo de mundos activados
    #[arg(long, default_value_t = 2)]
    max_worlds: usize,
    #[arg(long)]
    json: bool,
}

#[derive(Subcommand)]
enum SubsCommand {
    /// Genera el .subs desde el .ukj
    Extract {
        file: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long)]
        subs_dir: Option<PathBuf>,
        #[arg(long)]
        force: bool,
        #[arg(long)]
        json: bool,
    },
    /// Comprueba que el .subs corresponde al .ukj
    Verify {
        file: PathBuf,
        #[arg(long)]
        path: Option<PathBuf>,
        #[arg(long)]
        subs_dir: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Repara el .subs si falta, está corrupto o desincronizado
    Repair {
        file: PathBuf,
        #[arg(long)]
        path: Option<PathBuf>,
        #[arg(long)]
        subs_dir: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
}

fn print_json(value: &serde_json::Value) {
    match serde_json::to_string_pretty(value) {
        Ok(s) => println!("{s}"),
        Err(e) => eprintln!("error: no se pudo serializar la salida: {e}"),
    }
}

fn operator_config(normal: f32, anomalous: f32) -> OperatorConfig {
    OperatorConfig {
        stop_subparameter: true,
        continue_parameter: true,
        tics_normal: normal,
        tics_anomalous: anomalous,
        window_ms: 1000,
    }
}

fn log_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cabrita")
        .join("logs")
}

fn latest_log_path() -> Option<PathBuf> {
    let dir = log_dir();
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in std::fs::read_dir(&dir).ok()? {
        let entry = entry.ok()?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let modified = entry.metadata().ok()?.modified().ok()?;
        if best.as_ref().map(|(t, _)| modified > *t).unwrap_or(true) {
            best = Some((modified, path));
        }
    }
    best.map(|(_, path)| path)
}

fn format_ts(ms: u64) -> String {
    let s = ms / 1000;
    format!(
        "{:02}:{:02}:{:02}.{:03}",
        (s / 3600) % 24,
        (s / 60) % 60,
        s % 60,
        ms % 1000
    )
}

fn print_operator_log(path: &Path, json: bool) -> Result<(), String> {
    let events = read_log(path).map_err(|e| e.to_string())?;
    if json {
        print_json(&serde_json::json!({
            "path": path.display().to_string(),
            "events": events,
        }));
        return Ok(());
    }
    println!("log de operadores: {}", path.display());
    if events.is_empty() {
        println!("  (sin eventos)");
        return Ok(());
    }
    for e in events {
        let ts = e.get("ts_ms").and_then(|v| v.as_u64()).unwrap_or(0);
        let operator = e.get("operator").and_then(|v| v.as_str()).unwrap_or("?");
        let target = e.get("target").and_then(|v| v.as_str()).unwrap_or("?");
        let tics = e.get("tics_per_s").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let state = e.get("state").and_then(|v| v.as_str()).unwrap_or("?");
        let action = e.get("action").and_then(|v| v.as_str()).unwrap_or("");
        let rss = e.get("rss_bytes").and_then(|v| v.as_u64()).unwrap_or(0);
        println!(
            "  {}Z {operator} · {target} · {tics:.1} tics/s · {state} · {action} · RSS {}",
            format_ts(ts),
            format_bytes(rss as f64)
        );
    }
    Ok(())
}

fn write_demo_log(
    path: &Path,
    transitions: &[cabrita::operators::TicsTransition],
) -> Result<(), String> {
    let log = SessionLog::create(path).map_err(|e| e.to_string())?;
    for tr in transitions {
        let event = OperatorEvent {
            ts_ms: tr.ts_ms,
            operator: "stop-subparameter".to_string(),
            target: format!("blk.{}.ffn_*_exps", tr.layer),
            tics_per_s: tr.tics_per_s,
            state: tr.to.as_str().to_string(),
            action: format!("{} · 0 bytes", tr.action),
            rss_bytes: rss_bytes(),
        };
        log.write(&event).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn cmd_inspect(args: InspectArgs) -> ExitCode {
    let ukj = match UkjFile::open(&args.file, args.deep, true) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    let subs_status = if args.subs {
        match verify_subs(&args.file, None, args.subs_dir.as_deref()) {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(2);
            }
        }
    } else {
        None
    };
    let report = inspect_report(&ukj, args.deep, subs_status.as_ref());
    if args.json {
        print_json(&report);
    } else {
        print!("{}", format_summary(&report, args.tensors));
    }
    if args.log || args.log_file.is_some() {
        let path = match args.log_file.clone().or_else(|| {
            let near = SessionLog::path_for_model(&args.file);
            if near.is_file() {
                Some(near)
            } else {
                latest_log_path()
            }
        }) {
            Some(p) => p,
            None => {
                eprintln!(
                    "error: no hay logs de operadores en {}",
                    log_dir().display()
                );
                return ExitCode::from(2);
            }
        };
        if let Err(e) = print_operator_log(&path, args.json) {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    }
    ExitCode::SUCCESS
}

fn cmd_verify(args: VerifyArgs) -> ExitCode {
    let ukj = match UkjFile::open(&args.file, !args.quick, true) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        }
    };
    let subs_status = if args.subs {
        match verify_subs(&args.file, None, args.subs_dir.as_deref()) {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(1);
            }
        }
    } else {
        None
    };
    let report = verify_report(&ukj, !args.quick, subs_status.as_ref());
    if args.json {
        print_json(&report);
    } else {
        println!("OK · {}", ukj.path().display());
        println!(
            "model_id: {} · {} tensores",
            ukj.header.model.id,
            ukj.header.tensors.len()
        );
        println!("payload: {} bytes", ukj.container.payload_len);
        if args.quick {
            println!("payload sha256: omitido (--quick)");
        }
        if let Some(s) = &subs_status {
            println!(".subs: {} · {}", s.status, s.path);
            if !s.ok {
                println!("  {}", s.message);
            }
        }
    }
    let ok = report.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

fn cmd_subs(args: SubsArgs) -> ExitCode {
    match args.command {
        SubsCommand::Extract {
            file,
            out,
            subs_dir,
            force,
            json,
        } => match extract_subs(&file, out.as_deref(), subs_dir.as_deref(), force) {
            Ok(path) => {
                if json {
                    print_json(&serde_json::json!({
                        "ok": true,
                        "action": "extraído",
                        "path": path.display().to_string(),
                    }));
                } else {
                    println!("extraído · {}", path.display());
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::from(2)
            }
        },
        SubsCommand::Verify {
            file,
            path,
            subs_dir,
            json,
        } => match verify_subs(&file, path.as_deref(), subs_dir.as_deref()) {
            Ok(status) => {
                if json {
                    print_json(&serde_json::to_value(&status).unwrap_or_default());
                } else {
                    println!("{} · {}", status.status, status.path);
                    println!("{}", status.message);
                }
                if status.ok {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::from(1)
                }
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::from(2)
            }
        },
        SubsCommand::Repair {
            file,
            path,
            subs_dir,
            json,
        } => match ensure_subs(&file, path.as_deref(), subs_dir.as_deref()) {
            Ok(status) => {
                if json {
                    print_json(&serde_json::to_value(&status).unwrap_or_default());
                } else {
                    println!("{} · {}", status.status, status.path);
                    println!("{}", status.message);
                    if let Some(action) = &status.action {
                        println!("acción: {action}");
                    }
                }
                if status.ok {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::from(1)
                }
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::from(2)
            }
        },
    }
}

fn format_bytes(bytes: f64) -> String {
    if bytes >= 1024.0 * 1024.0 * 1024.0 {
        format!("{:.2} GB", bytes / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024.0 * 1024.0 {
        format!("{:.1} MB", bytes / (1024.0 * 1024.0))
    } else if bytes >= 1024.0 {
        format!("{:.1} KB", bytes / 1024.0)
    } else {
        format!("{bytes:.0} B")
    }
}

fn format_signed_bytes(bytes: i64) -> String {
    if bytes < 0 {
        format!("-{}", format_bytes((-bytes) as f64))
    } else {
        format!("+{}", format_bytes(bytes as f64))
    }
}

fn format_metrics(m: &TurnMetrics) -> String {
    format!(
        "[métricas] densidad={} · prompt={} tok · generados={} tok · \
         prefill={:.0} ms · decode={:.2} tok/s · ttft={:.0} ms · latencia={:.2} s · \
         RSS={} · pico={} · KV={} · ctx={}/{} · stop={}",
        m.density,
        m.prompt_tokens,
        m.generated_tokens,
        m.prefill_ms,
        m.tokens_per_s,
        m.ttft_ms,
        m.latency_ms / 1000.0,
        format_bytes(m.rss_mb * 1024.0 * 1024.0),
        format_bytes(m.peak_rss_mb * 1024.0 * 1024.0),
        format_bytes(m.kv_cache_mb * 1024.0 * 1024.0),
        m.ctx_used,
        m.ctx_capacity,
        m.stop
    )
}

fn gen_options(args: &ChatArgs) -> GenOptions {
    GenOptions {
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repetition_penalty: args.repetition_penalty,
        seed: args.seed,
        stream: !args.no_stream,
    }
}

fn open_session(
    file: &PathBuf,
    ctx: usize,
    system: Option<String>,
    verify_payload: bool,
) -> Result<ChatSession, String> {
    let model = if verify_payload {
        OlmoeModel::open_verified(file)
    } else {
        OlmoeModel::open(file)
    }
    .map_err(|e| e.to_string())?;
    ChatSession::new(model, ctx, system).map_err(|e| e.to_string())
}

struct LiveMeter {
    start: std::time::Instant,
    count: usize,
    enabled: bool,
}

impl LiveMeter {
    fn new(enabled: bool) -> Self {
        Self {
            start: std::time::Instant::now(),
            count: 0,
            enabled,
        }
    }

    fn tick(&mut self) {
        self.count += 1;
        if self.enabled && self.count % 8 == 0 {
            let s = self.start.elapsed().as_secs_f64().max(1e-6);
            eprint!(
                "\r[en vivo] {} tok · {:.2} tok/s",
                self.count,
                self.count as f64 / s
            );
            io::stderr().flush().ok();
        }
    }

    fn finish(&self) {
        if self.enabled && self.count >= 8 {
            eprintln!();
        }
    }
}

fn find_taxonomy_path(explicit: Option<&Path>) -> Result<PathBuf, String> {
    if let Some(p) = explicit {
        return Ok(p.to_path_buf());
    }
    [
        "worlds/olmoe.json",
        "../worlds/olmoe.json",
        "../../worlds/olmoe.json",
    ]
    .iter()
    .map(PathBuf::from)
    .find(|p| p.is_file())
    .ok_or_else(|| "no se encontró worlds/olmoe.json; pasa --worlds o usa --world".to_string())
}

/// Router por defecto de `chat`: elige el mundo a partir del texto del usuario.
struct ChatRouter {
    taxonomy: WorldTaxonomy,
    router: TinyRouter,
    policy: ActivationPolicy,
}

impl ChatRouter {
    fn load(explicit: Option<&Path>) -> Result<Self, String> {
        let path = find_taxonomy_path(explicit)?;
        let taxonomy = WorldTaxonomy::load(&path).map_err(|e| e.to_string())?;
        let router = TinyRouter::train(&taxonomy).map_err(|e| e.to_string())?;
        Ok(Self {
            taxonomy,
            router,
            policy: ActivationPolicy::default(),
        })
    }

    /// Aplica la máscara del mundo elegido. Devuelve la activación y el máximo
    /// de expertos por capa que queda habilitado.
    fn apply(&self, session: &mut ChatSession, text: &str) -> Result<(Activation, usize), String> {
        let activation = self.router.activate(&self.taxonomy, text, &self.policy);
        let cfg = session.model.config().clone();
        if activation.incomplete {
            session
                .model
                .set_expert_mask(None)
                .map_err(|e| e.to_string())?;
            return Ok((activation, cfg.experts));
        }
        let ids: Vec<String> = activation.worlds.iter().map(|w| w.id.clone()).collect();
        let mask = cabrita::routes::expert_mask_for(&self.taxonomy, &ids, cfg.layers, cfg.experts)
            .map_err(|e| e.to_string())?;
        let experts = mask
            .iter()
            .map(|bits| (0..cfg.experts).filter(|e| bits & (1u64 << e) != 0).count())
            .max()
            .unwrap_or(0);
        session
            .model
            .set_expert_mask(Some(mask))
            .map_err(|e| e.to_string())?;
        Ok((activation, experts))
    }
}

fn print_activation(activation: &Activation, experts: usize) {
    if activation.incomplete {
        eprintln!(
            "router: modo incompleto · {} · modelo base completo",
            activation.warning.as_deref().unwrap_or("sin mundo")
        );
        return;
    }
    let worlds: Vec<String> = activation
        .worlds
        .iter()
        .map(|w| format!("{} {:.0}%", w.label, w.probability * 100.0))
        .collect();
    eprintln!("router: {} · {experts} expertos/capa", worlds.join(" + "));
}

fn cmd_chat(args: ChatArgs) -> ExitCode {
    let verify = !args.no_verify_payload;
    if verify {
        eprintln!("verificando payload y trailer…");
    } else {
        eprintln!("aviso: payload no verificado (--no-verify-payload)");
    }
    let mut session = match open_session(&args.file, args.ctx, args.system.clone(), verify) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    if args.backend == "vulkan" {
        match session.model.enable_vulkan(args.vulkan_cache_mb) {
            Ok(info) => eprintln!(
                "backend: Vulkan · {} · {} · caché {} MB",
                info.name, info.device_type, args.vulkan_cache_mb
            ),
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(2);
            }
        }
    }
    if args.mask_all {
        let layers = session.model.config().layers;
        if let Err(e) = session.model.set_expert_mask(Some(vec![u64::MAX; layers])) {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
        eprintln!("máscara: 64 expertos por capa");
    } else if !args.world.is_empty() {
        let taxonomy = match find_taxonomy_path(args.worlds.as_deref())
            .and_then(|path| WorldTaxonomy::load(&path).map_err(|e| e.to_string()))
        {
            Ok(t) => t,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(2);
            }
        };
        let cfg = session.model.config().clone();
        let mask =
            match cabrita::routes::expert_mask_for(&taxonomy, &args.world, cfg.layers, cfg.experts)
            {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
        if let Err(e) = session.model.set_expert_mask(Some(mask)) {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
        let bits = session
            .model
            .expert_mask()
            .and_then(|m| m.first())
            .copied()
            .unwrap_or(0);
        let n = (0..64).filter(|e| bits & (1u64 << e) != 0).count();
        eprintln!(
            "mundo forzado: {} · {n} expertos por capa",
            args.world.join(", ")
        );
    }

    // Sin --world ni --mask-all, el router elige el mundo de cada prompt.
    let router = if args.mask_all || !args.world.is_empty() {
        None
    } else {
        match ChatRouter::load(args.worlds.as_deref()) {
            Ok(r) => Some(r),
            Err(e) if args.worlds.is_some() => {
                eprintln!("error: {e}");
                return ExitCode::from(2);
            }
            Err(e) => {
                eprintln!("aviso: router no disponible ({e}); se usa el modelo completo");
                None
            }
        }
    };
    if args.no_renorm {
        session.model.set_renormalize_moe(false);
        eprintln!("renormalización MoE desactivada");
    }
    let mut operator_log: Option<PathBuf> = None;
    if args.tics || args.log || args.log_file.is_some() {
        let path = args
            .log_file
            .clone()
            .unwrap_or_else(|| SessionLog::path_for_model(&args.file));
        let cfg = operator_config(args.tics_normal, args.tics_anomalous);
        if let Err(e) = cfg.validate() {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
        if let Err(e) = session.model.start_tics(cfg.clone(), Some(path.clone())) {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
        eprintln!(
            "tics: normal {}/s · anómalo {}/s · log {}",
            cfg.tics_normal,
            cfg.tics_anomalous,
            path.display()
        );
        operator_log = Some(path);
    }
    let opts = gen_options(&args);

    if let Some(prompt) = &args.prompt {
        let mut router_info = None;
        if let Some(r) = &router {
            match r.apply(&mut session, prompt) {
                Ok((activation, experts)) => {
                    print_activation(&activation, experts);
                    router_info = Some((activation, experts));
                }
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            }
        }
        let mut printed = false;
        let mut meter = LiveMeter::new(!args.no_stream && !args.json);
        let result = session.respond(prompt, &opts, |piece| {
            print!("{piece}");
            io::stdout().flush().ok();
            printed = true;
            meter.tick();
        });
        meter.finish();
        match result {
            Ok((response, metrics)) => {
                if args.no_stream && !args.json {
                    println!("{response}");
                } else if printed {
                    println!();
                }
                if args.json {
                    print_json(&serde_json::json!({
                        "response": response,
                        "metrics": metrics,
                        "tics_actions": session.model.tics_actions(),
                        "operator_log": operator_log.as_ref().map(|p| p.display().to_string()),
                        "backend": args.backend,
                        "vulkan_cached_bytes": session.model.vulkan_cached_bytes(),
                        "router": router_info.as_ref().map(|(a, n)| serde_json::json!({
                            "worlds": &a.worlds,
                            "incomplete": a.incomplete,
                            "warning": &a.warning,
                            "experts_per_layer": n,
                        })),
                    }));
                } else if !args.no_metrics {
                    eprintln!("{}", format_metrics(&metrics));
                }
                if !args.json {
                    if let Some(path) = &operator_log {
                        eprintln!(
                            "tics: {} acciones · log {}",
                            session.model.tics_actions(),
                            path.display()
                        );
                    }
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::from(2)
            }
        }
    } else {
        repl(
            &mut session,
            &opts,
            args.json,
            args.no_metrics,
            operator_log.as_deref(),
            router.as_ref(),
        )
    }
}

fn repl(
    session: &mut ChatSession,
    opts: &GenOptions,
    json: bool,
    no_metrics: bool,
    operator_log: Option<&Path>,
    router: Option<&ChatRouter>,
) -> ExitCode {
    println!("Cabrita · escribe /exit para salir, /reset para reiniciar, /system <texto>");
    let stdin = io::stdin();
    loop {
        print!("tú> ");
        io::stdout().flush().ok();
        let mut line = String::new();
        match stdin.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => {
                eprintln!("error leyendo stdin: {e}");
                return ExitCode::from(2);
            }
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "/exit" || line == "/quit" {
            break;
        }
        if line == "/reset" {
            session.reset();
            println!("sesión reiniciada");
            continue;
        }
        if let Some(sys) = line.strip_prefix("/system ") {
            session.set_system(Some(sys.to_string()));
            println!("system prompt actualizado");
            continue;
        }

        let mut router_info = None;
        if let Some(r) = router {
            match r.apply(session, line) {
                Ok((activation, experts)) => {
                    print_activation(&activation, experts);
                    router_info = Some((activation, experts));
                }
                Err(e) => {
                    eprintln!("error: {e}");
                    continue;
                }
            }
        }

        print!("cabrita> ");
        io::stdout().flush().ok();
        let mut meter = LiveMeter::new(!json);
        let result = session.respond(line, opts, |piece| {
            print!("{piece}");
            io::stdout().flush().ok();
            meter.tick();
        });
        meter.finish();
        match result {
            Ok((response, metrics)) => {
                println!();
                if json {
                    print_json(&serde_json::json!({
                        "response": response,
                        "metrics": metrics,
                        "router": router_info.as_ref().map(|(a, n)| serde_json::json!({
                            "worlds": &a.worlds,
                            "incomplete": a.incomplete,
                            "warning": &a.warning,
                            "experts_per_layer": n,
                        })),
                    }));
                } else if !no_metrics {
                    eprintln!("{}", format_metrics(&metrics));
                }
            }
            Err(e) => {
                eprintln!("\nerror: {e}");
            }
        }
    }
    if let Some(path) = operator_log {
        eprintln!(
            "tics: {} acciones · log {}",
            session.model.tics_actions(),
            path.display()
        );
    }
    ExitCode::SUCCESS
}

fn cmd_bench(args: BenchArgs) -> ExitCode {
    let start = std::time::Instant::now();
    let mut session = match open_session(&args.file, args.ctx, None, false) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    if args.backend == "vulkan" {
        match session.model.enable_vulkan(args.vulkan_cache_mb) {
            Ok(info) => eprintln!(
                "backend: Vulkan · {} · {} · caché {} MB",
                info.name, info.device_type, args.vulkan_cache_mb
            ),
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(2);
            }
        }
    }
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;
    let opts = GenOptions {
        max_tokens: args.tokens,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: 42,
        stream: false,
    };
    let result = session.respond(&args.prompt, &opts, |_| {});
    match result {
        Ok((response, metrics)) => {
            if args.json {
                print_json(&serde_json::json!({
                    "load_ms": load_ms,
                    "response": response,
                    "metrics": metrics,
                    "backend": args.backend,
                    "vulkan_cached_bytes": session.model.vulkan_cached_bytes(),
                }));
            } else {
                println!("carga: {load_ms:.0} ms");
                println!("respuesta: {response}");
                println!("{}", format_metrics(&metrics));
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}

fn cmd_tics(args: TicsArgs) -> ExitCode {
    let cfg = operator_config(args.normal, args.anomalous);
    if let Err(e) = cfg.validate() {
        eprintln!("error: {e}");
        return ExitCode::from(2);
    }
    let log_path = if args.log || args.log_file.is_some() {
        Some(args.log_file.clone().unwrap_or_else(|| match &args.model {
            Some(model) => SessionLog::path_for_model(model),
            None => SessionLog::default_path(),
        }))
    } else {
        None
    };

    if let Some(model_path) = &args.model {
        let mut session = match open_session(model_path, args.ctx, None, false) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(2);
            }
        };
        if args.backend == "vulkan" {
            match session.model.enable_vulkan(args.vulkan_cache_mb) {
                Ok(info) => eprintln!(
                    "backend: Vulkan · {} · {} · caché {} MB",
                    info.name, info.device_type, args.vulkan_cache_mb
                ),
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            }
        }
        if let Err(e) = session.model.start_tics(cfg.clone(), log_path.clone()) {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
        let opts = GenOptions {
            max_tokens: args.max_tokens,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: 42,
            stream: false,
        };
        let result = session.respond(&args.prompt, &opts, |_| {});
        let (response, metrics) = match result {
            Ok(v) => v,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(2);
            }
        };
        let tracker = session.model.take_tics();
        let transitions = tracker.map(|t| t.into_transitions()).unwrap_or_default();
        let actions = session.model.tics_actions();
        if args.json {
            print_json(&serde_json::json!({
                "config": cfg,
                "response": response,
                "metrics": metrics,
                "actions": actions,
                "transitions": transitions,
                "log": log_path.as_ref().map(|p| p.display().to_string()),
            }));
        } else {
            println!("{response}");
            println!("{}", format_metrics(&metrics));
            println!(
                "tics: {actions} acciones · {} transiciones",
                transitions.len()
            );
            for tr in &transitions {
                println!(
                    "  capa {} {} → {} · {:.1} tics/s · {}",
                    tr.layer,
                    tr.from.as_str(),
                    tr.to.as_str(),
                    tr.tics_per_s,
                    tr.action
                );
            }
            if let Some(path) = &log_path {
                println!("log: {}", path.display());
            }
        }
        return ExitCode::SUCCESS;
    }

    // Demostración sintética: 1/s (normal), 8/s (histéresis), 25/s (anómalo)
    // y después inactividad para forzar el offload.
    let mut tracker = match TicsTracker::new(cfg.clone(), 1) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    let start = Instant::now();
    let mut samples = Vec::new();
    for second in 0..8u64 {
        let count = match second {
            0 | 1 => 1,
            2 | 3 => 8,
            4 | 5 => 25,
            _ => 0,
        };
        let now = start + Duration::from_secs(second);
        let transition = if count > 0 {
            tracker.record(0, count, now)
        } else {
            tracker.tick(0, now)
        };
        samples.push((second, count, tracker.rate(0, now), tracker.state(0)));
        let _ = transition;
    }
    let transitions = tracker.transitions().to_vec();
    if let Some(path) = &log_path {
        if let Err(e) = write_demo_log(path, &transitions) {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    }
    if args.json {
        print_json(&serde_json::json!({
            "config": cfg,
            "samples": samples.iter().map(|(s, c, r, st)| serde_json::json!({
                "second": s,
                "activations": c,
                "tics_per_s": r,
                "state": st.as_str(),
            })).collect::<Vec<_>>(),
            "transitions": transitions,
            "log": log_path.as_ref().map(|p| p.display().to_string()),
        }));
    } else {
        println!(
            "tics: normal {}/s · anómalo {}/s · ventana {} ms",
            cfg.tics_normal, cfg.tics_anomalous, cfg.window_ms
        );
        for (second, count, rate, state) in &samples {
            println!(
                "  t={second}s · {count:>2} activaciones · {rate:>5.1}/s · {}",
                state.as_str()
            );
        }
        println!("transiciones: {}", transitions.len());
        for tr in &transitions {
            println!(
                "  capa {} {} → {} · {:.1} tics/s · {}",
                tr.layer,
                tr.from.as_str(),
                tr.to.as_str(),
                tr.tics_per_s,
                tr.action
            );
        }
        if let Some(path) = &log_path {
            println!("log: {}", path.display());
        }
    }
    ExitCode::SUCCESS
}

fn cmd_vulkan(args: VulkanArgs) -> ExitCode {
    if args.selftest {
        match cabrita::vulkan::selftest() {
            Ok((info, rows)) => {
                if args.json {
                    print_json(&serde_json::json!({
                        "adapter": info,
                        "checks": rows,
                    }));
                } else {
                    println!(
                        "Vulkan: {} · {} · {}",
                        info.name, info.backend, info.device_type
                    );
                    for row in &rows {
                        println!(
                            "  {:<4} {}x{} · max_abs={:.3e} · max_rel={:.3e} · {}",
                            row.quant,
                            row.rows,
                            row.cols,
                            row.max_abs_diff,
                            row.max_rel_diff,
                            if row.ok { "OK" } else { "FALLA" }
                        );
                    }
                }
                if rows.iter().all(|r| r.ok) {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::from(1)
                }
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::from(2)
            }
        }
    } else {
        match cabrita::vulkan::probe() {
            Ok(info) => {
                if args.json {
                    print_json(&serde_json::to_value(&info).unwrap_or_default());
                } else {
                    println!("adaptador: {}", info.name);
                    println!(
                        "backend: {} · tipo: {} · driver: {} {}",
                        info.backend, info.device_type, info.driver, info.driver_info
                    );
                    println!(
                        "límites: storage {:.1} MB · buffer {:.1} MB",
                        info.max_storage_buffer as f64 / (1024.0 * 1024.0),
                        info.max_buffer as f64 / (1024.0 * 1024.0)
                    );
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::from(2)
            }
        }
    }
}

fn cmd_route(args: RouteArgs) -> ExitCode {
    let taxonomy = match WorldTaxonomy::load(&args.worlds) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    let policy = ActivationPolicy {
        threshold: args.threshold,
        max_worlds: args.max_worlds,
        ..Default::default()
    };
    let router = match TinyRouter::train(&taxonomy) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };

    let eval = if args.eval {
        match evaluate_holdout(&taxonomy, args.holdout_every, &policy) {
            Ok(r) => Some(r),
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(2);
            }
        }
    } else {
        None
    };

    let mut activation = None;
    let mut plan_summary = None;
    let mut load_stats: Option<LoadStats> = None;
    let mut unload_stats: Option<LoadStats> = None;

    if let Some(text) = &args.text {
        let act = router.activate(&taxonomy, text, &policy);
        if args.load || args.unload {
            let model = match OlmoeModel::open(&args.file) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            let full_bytes = model.payload_len();
            let plan = match build_plan(
                &taxonomy,
                &act,
                &model.ukj.header.routes,
                model.tensor_infos(),
                model.config().experts,
                full_bytes,
            ) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            if let Some(mask) = &plan.expert_mask {
                let bits = mask.first().copied().unwrap_or(0);
                let n = (0..64).filter(|e| bits & (1u64 << e) != 0).count();
                plan_summary = Some(serde_json::json!({
                    "worlds": plan.worlds.clone(),
                    "routes": plan.routes.clone(),
                    "tensors": plan.tensors.len(),
                    "ranges": plan.ranges.len(),
                    "loaded_bytes": plan.loaded_bytes,
                    "full_bytes": plan.full_bytes,
                    "ratio": plan.ratio(),
                    "expert_mask_layers": mask.len(),
                    "experts_per_layer": n,
                    "incomplete": plan.incomplete,
                    "warning": plan.warning.clone(),
                }));
            } else {
                plan_summary = Some(serde_json::json!({
                    "worlds": plan.worlds.clone(),
                    "routes": plan.routes.clone(),
                    "tensors": plan.tensors.len(),
                    "ranges": plan.ranges.len(),
                    "loaded_bytes": plan.loaded_bytes,
                    "full_bytes": plan.full_bytes,
                    "ratio": plan.ratio(),
                    "incomplete": plan.incomplete,
                    "warning": plan.warning.clone(),
                }));
            }
            load_stats = Some(model.load_plan(&plan));
            if args.unload {
                unload_stats = Some(model.unload_plan(&plan));
            }
        }
        activation = Some(act);
    }

    if args.json {
        let value = serde_json::json!({
            "taxonomy": {
                "worlds": taxonomy.worlds.len(),
                "feature_dim": taxonomy.feature_dim,
            },
            "eval": eval,
            "activation": activation,
            "plan": plan_summary,
            "load": load_stats,
            "unload": unload_stats,
        });
        print_json(&value);
        return ExitCode::SUCCESS;
    }

    if let Some(report) = &eval {
        println!(
            "eval: {}/{} correctos ({:.1}%) · train {} · test {}",
            report.correct,
            report.test_examples,
            report.accuracy * 100.0,
            report.train_examples,
            report.test_examples
        );
        for (world, (ok, total)) in &report.per_world {
            println!("  {world:14} {ok}/{total}");
        }
        for (expected, got, text) in &report.errors {
            println!("  error: esperado {expected}, obtenido {got} · {text:?}");
        }
    }

    if let Some(act) = &activation {
        if let Some(text) = &args.text {
            println!("consulta: {text:?}");
        }
        if act.incomplete {
            println!(
                "modo incompleto: {}",
                act.warning.clone().unwrap_or_default()
            );
        } else {
            for w in &act.worlds {
                println!(
                    "mundo: {:<12} {:<14} prioridad {:>2} · p={:.1}%{}",
                    w.id,
                    w.label,
                    w.priority,
                    w.probability * 100.0,
                    if w.provisional { " · provisional" } else { "" }
                );
            }
            if let Some(w) = &act.warning {
                println!("aviso: {w}");
            }
        }
        println!(
            "candidatos: {}",
            act.probabilities
                .iter()
                .take(4)
                .map(|(id, p)| format!("{id} {:.0}%", p * 100.0))
                .collect::<Vec<_>>()
                .join(" · ")
        );
    }

    if let Some(plan) = &plan_summary {
        println!(
            "plan: {} tensores · {} rangos · {} de {} ({:.1}%){}",
            plan["tensors"],
            plan["ranges"],
            format_bytes(plan["loaded_bytes"].as_f64().unwrap_or(0.0)),
            format_bytes(plan["full_bytes"].as_f64().unwrap_or(0.0)),
            plan["ratio"].as_f64().unwrap_or(0.0) * 100.0,
            plan["experts_per_layer"]
                .as_u64()
                .map(|n| format!(" · {n} expertos por capa"))
                .unwrap_or_default()
        );
    }
    if let Some(load) = &load_stats {
        println!(
            "carga: {} · {} páginas · RSS {} → {} (Δ {}) · {:.0} ms",
            format_bytes(load.bytes as f64),
            load.pages,
            format_bytes(load.rss_before as f64),
            format_bytes(load.rss_after as f64),
            format_signed_bytes(load.rss_delta),
            load.elapsed_ms
        );
    }
    if let Some(unload) = &unload_stats {
        println!(
            "descarga: RSS {} → {} (Δ {}) · {:.0} ms",
            format_bytes(unload.rss_before as f64),
            format_bytes(unload.rss_after as f64),
            format_signed_bytes(unload.rss_delta),
            unload.elapsed_ms
        );
    }
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Inspect(args) => cmd_inspect(args),
        Command::Verify(args) => cmd_verify(args),
        Command::Subs(args) => cmd_subs(args),
        Command::Chat(args) => cmd_chat(args),
        Command::Bench(args) => cmd_bench(args),
        Command::Route(args) => cmd_route(args),
        Command::Tics(args) => cmd_tics(args),
        Command::Vulkan(args) => cmd_vulkan(args),
    }
}
