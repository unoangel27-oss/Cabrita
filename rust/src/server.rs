//! Servidor HTTP compatible OpenAI y GUI embebida de Cabrita.
//!
//! Modelo de concurrencia: un solo hilo atiende una conexión detrás de otra. El
//! modelo usa `RefCell` y no necesita sincronización, pero una petición lenta
//! retrasa a la siguiente; por eso cada socket lleva un tiempo máximo de espera
//! ([`IO_TIMEOUT`]) que cierra al cliente ocioso o atascado. Para servir a varios
//! clientes a la vez hay que levantar varias instancias.
//!
//! Seguridad: con `--api-key`, tanto `/v1/*` como `/api/*` exigen
//! `Authorization: Bearer`; solo `/` y `/health` quedan abiertos. CORS se limita
//! al origen propio (`scheme://Host`), así que una página de otro origen no
//! puede leer las respuestas ni superar el preflight de los POST. La GUI recibe
//! la clave inyectada en su propio HTML para poder llamar a la API.
//!
//! La GUI incluye un editor mínimo de mundos que lista la taxonomía, abre un
//! mundo y despliega sus pesos con `madvise`.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use serde_json::{json, Value};

use crate::chat::{ChatSession, GenOptions};
use crate::error::{Error, Result};
use crate::model::rss_bytes;
use crate::router::{ActivatedWorld, Activation, TinyRouter, WorldTaxonomy};
use crate::routes::{build_plan, expert_mask_for};

const MAX_BODY: usize = 8 * 1024 * 1024;
/// Espera máxima por lectura o escritura de un socket: un cliente que abre TCP
/// y no envía nada no bloquea el servidor indefinidamente.
const IO_TIMEOUT: Duration = Duration::from_secs(30);
const INDEX_HTML: &str = r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cabrita</title>
<style>
:root{color-scheme:dark;--bg:#101216;--panel:#181c22;--line:#2a303a;--text:#e8edf5;--muted:#9aa6b8;--accent:#e0a458}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--text)}
header{padding:18px 24px;border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:baseline}
h1{font-size:20px;margin:0}h2{font-size:16px;margin:0 0 10px}
main{display:grid;grid-template-columns:320px 1fr;gap:18px;padding:18px;max-width:1200px;margin:auto}
section{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}
.muted{color:var(--muted);font-size:13px}.world{border:1px solid var(--line);border-radius:8px;padding:10px;margin-bottom:8px}
.world strong{display:block}.row{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
button{background:#252b35;color:var(--text);border:1px solid var(--line);border-radius:7px;padding:6px 10px;cursor:pointer}
button:hover{border-color:var(--accent)}button.primary{background:var(--accent);color:#1a1206;border-color:var(--accent);font-weight:600}
#chat{height:420px;overflow:auto;border:1px solid var(--line);border-radius:8px;padding:12px;margin-bottom:10px;background:#12151a}
.msg{margin:0 0 10px;white-space:pre-wrap}.msg.user{color:#b9d4ff}.msg.assistant{color:var(--text)}
textarea{width:100%;min-height:74px;background:#12151a;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:10px;resize:vertical}
pre{white-space:pre-wrap;font-size:12px;color:var(--muted);margin:8px 0 0}
@media(max-width:800px){main{grid-template-columns:1fr}}
</style>
</head>
<body>
<header><h1>Cabrita</h1><span class="muted" id="status">loading…</span></header>
<main>
<section>
  <h2>Worlds</h2>
  <p class="muted">Click open to inspect, then deploy to load its expert ranges.</p>
  <div id="worlds"></div>
</section>
<section>
  <h2>Chat</h2>
  <div id="chat"></div>
  <textarea id="prompt" placeholder="Write a message…"></textarea>
  <div class="row"><button class="primary" id="send">Send</button><button id="clear">Clear</button></div>
  <pre id="meta"></pre>
</section>
</main>
<script>
const $ = (id) => document.getElementById(id);
const API_KEY = __CABRITA_API_KEY__;
const esc = (s) => String(s).replace(/[<>&"']/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;',"'":'&#39;'}[c]));
let history = [];
async function j(url, opts){
  opts = opts || {};
  const headers = Object.assign({}, opts.headers);
  if(API_KEY) headers.authorization = 'Bearer ' + API_KEY;
  const r = await fetch(url, Object.assign({}, opts, {headers}));
  if(!r.ok) throw new Error(await r.text());
  return r.json();
}
async function load(){
  const s=await j('/api/status'); $('status').textContent=`${esc(s.model)} · ${esc(s.density)} · ${esc(s.ctx)} ctx · ${esc(s.rss_mb.toFixed(0))} MB`;
  const w=await j('/api/worlds'); const box=$('worlds'); box.innerHTML='';
  if(!w.worlds.length){box.innerHTML='<p class="muted">No taxonomy loaded.</p>';return}
  for(const world of w.worlds){
    const el=document.createElement('div'); el.className='world';
    el.innerHTML=`<strong>${esc(world.label||world.id)}</strong><span class="muted">${esc(world.id)} · priority ${esc(world.priority)}${world.provisional?' · provisional':''}</span>`;
    const row=document.createElement('div'); row.className='row';
    const open=document.createElement('button'); open.textContent='Open';
    open.onclick=()=>{el.querySelector('pre')?.remove();const p=document.createElement('pre');p.textContent=JSON.stringify(world,null,2);el.appendChild(p)};
    const deploy=document.createElement('button'); deploy.className='primary'; deploy.textContent='Deploy weights';
    deploy.onclick=async()=>{p.textContent='deploying…';try{const r=await j('/api/worlds/deploy',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({worlds:[world.id]})});p.textContent=JSON.stringify(r,null,2)}catch(e){p.textContent=String(e)}};
    const p=document.createElement('pre');
    row.append(open,deploy); el.append(row,p); box.append(el);
  }
}
async function send(){
  const text=$('prompt').value.trim(); if(!text)return; $('prompt').value='';
  history.push({role:'user',content:text}); render();
  $('send').disabled=true;
  try{
    const r=await j('/v1/chat/completions',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({model:'cabrita',messages:history,max_tokens:128})});
    const content=r.choices[0].message.content; history.push({role:'assistant',content}); render();
    $('meta').textContent=JSON.stringify(r.usage,null,2);
  }catch(e){$('meta').textContent=String(e)}
  $('send').disabled=false;
}
function render(){ $('chat').innerHTML=history.map(m=>`<p class="msg ${esc(m.role)}"><b>${esc(m.role)}:</b> ${esc(m.content)}</p>`).join(''); $('chat').scrollTop=$('chat').scrollHeight }
$('send').onclick=send; $('clear').onclick=()=>{history=[];render();$('meta').textContent=''};
$('prompt').addEventListener('keydown',e=>{if(e.key==='Enter'&&(e.ctrlKey||e.metaKey))send()});
load().catch(e=>$('status').textContent=String(e));
</script>
</body>
</html>"##;

struct Request {
    method: String,
    path: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

struct ServerState {
    session: ChatSession,
    model_name: String,
    taxonomy: Option<WorldTaxonomy>,
    api_key: Option<String>,
    tls: bool,
    started: Instant,
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    messages: Vec<WireMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct WireMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct DeployRequest {
    worlds: Vec<String>,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn find_taxonomy(explicit: Option<&Path>) -> Option<WorldTaxonomy> {
    if let Some(path) = explicit {
        return WorldTaxonomy::load(path).ok();
    }
    for candidate in [
        "worlds/olmoe.json",
        "../worlds/olmoe.json",
        "../../worlds/olmoe.json",
    ] {
        let path = PathBuf::from(candidate);
        if path.is_file() {
            return WorldTaxonomy::load(&path).ok();
        }
    }
    None
}

/// Lee del socket tratando un timeout como conexión ociosa: no hay petición
/// que responder y el cliente se cierra sin bloquear el servidor.
fn read_chunk<R: Read>(stream: &mut R, chunk: &mut [u8]) -> Result<Option<usize>> {
    match stream.read(chunk) {
        Ok(n) => Ok(Some(n)),
        Err(e)
            if matches!(
                e.kind(),
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
            ) =>
        {
            Ok(None)
        }
        Err(e) => Err(e.into()),
    }
}

fn read_request<S: Read>(stream: &mut S) -> Result<Option<Request>> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    let header_end = loop {
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
        if buf.len() > MAX_BODY {
            return Err(Error::new("cabecera HTTP demasiado grande"));
        }
        let Some(n) = read_chunk(stream, &mut chunk)? else {
            return Ok(None);
        };
        if n == 0 {
            return Ok(None);
        }
        buf.extend_from_slice(&chunk[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..header_end - 4]);
    let mut lines = head.lines();
    let request_line = lines
        .next()
        .ok_or_else(|| Error::new("petición HTTP vacía"))?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let raw_path = parts.next().unwrap_or("/");
    let path = raw_path.split('?').next().unwrap_or("/").to_string();
    let mut headers = HashMap::new();
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }
    let len = headers
        .get("content-length")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    if len > MAX_BODY {
        return Err(Error::new("cuerpo HTTP demasiado grande"));
    }
    while buf.len() < header_end + len {
        let Some(n) = read_chunk(stream, &mut chunk)? else {
            return Err(Error::new(
                "cuerpo HTTP truncado o tiempo de espera agotado",
            ));
        };
        if n == 0 {
            return Err(Error::new("cuerpo HTTP truncado"));
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    let body = buf[header_end..header_end + len].to_vec();
    Ok(Some(Request {
        method,
        path,
        headers,
        body,
    }))
}

fn write_headers<W: Write>(
    out: &mut W,
    cors: Option<&str>,
    status: &str,
    content_type: &str,
    extra: &[(&str, &str)],
) -> Result<()> {
    write!(out, "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\n")?;
    if let Some(origin) = cors {
        write!(
            out,
            "Access-Control-Allow-Origin: {origin}\r\n\
             Access-Control-Allow-Headers: content-type, authorization\r\n\
             Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
             Vary: Origin\r\n"
        )?;
    }
    write!(out, "Connection: close\r\n")?;
    for (k, v) in extra {
        write!(out, "{k}: {v}\r\n")?;
    }
    write!(out, "\r\n")?;
    Ok(())
}

fn write_json<S: Write>(
    stream: &mut S,
    cors: Option<&str>,
    status: &str,
    value: &Value,
) -> Result<()> {
    let body = serde_json::to_vec(value)?;
    write_headers(
        stream,
        cors,
        status,
        "application/json; charset=utf-8",
        &[("Content-Length", &body.len().to_string())],
    )?;
    stream.write_all(&body)?;
    stream.flush()?;
    Ok(())
}

fn json_error<S: Write>(
    stream: &mut S,
    cors: Option<&str>,
    status: &str,
    message: &str,
) -> Result<()> {
    write_json(
        stream,
        cors,
        status,
        &json!({"error": {"message": message}}),
    )
}

/// Rutas que exigen clave cuando `--api-key` está activo. Solo la GUI (`/`) y
/// `/health` quedan abiertos; `/v1/*` y `/api/*` son datos y acciones del modelo.
fn needs_auth(path: &str) -> bool {
    path.starts_with("/v1/") || path.starts_with("/api/") || path == "/v1" || path == "/api"
}

/// CORS solo para el origen que sirve esta misma instancia: `Origin` tiene que
/// coincidir con `scheme://Host`. Cualquier otro origen se queda sin cabeceras,
/// de modo que no puede leer respuestas ni superar el preflight de los POST.
fn cors_origin(req: &Request, tls: bool) -> Option<String> {
    let origin = req.headers.get("origin")?;
    if origin.contains('\r') || origin.contains('\n') {
        return None;
    }
    let host = req.headers.get("host")?;
    let scheme = if tls { "https" } else { "http" };
    let expected = format!("{scheme}://{host}");
    if origin.trim_end_matches('/').eq_ignore_ascii_case(&expected) {
        Some(origin.clone())
    } else {
        None
    }
}

/// GUI con la clave inyectada (si hay) para que sus `fetch` autentiquen.
fn index_html(state: &ServerState) -> String {
    let literal = serde_json::to_string(state.api_key.as_deref().unwrap_or(""))
        .unwrap_or_else(|_| "\"\"".to_string());
    INDEX_HTML.replace("__CABRITA_API_KEY__", &literal)
}

fn authorized(req: &Request, state: &ServerState) -> bool {
    match &state.api_key {
        None => true,
        Some(expected) => req
            .headers
            .get("authorization")
            .and_then(|v| v.strip_prefix("Bearer "))
            .map(|v| v == expected)
            .unwrap_or(false),
    }
}

fn messages(req: &ChatRequest) -> Vec<crate::tokenizer::ChatMessage> {
    req.messages
        .iter()
        .map(|m| crate::tokenizer::ChatMessage {
            role: match m.role.as_str() {
                "system" => crate::tokenizer::Role::System,
                "assistant" => crate::tokenizer::Role::Assistant,
                _ => crate::tokenizer::Role::User,
            },
            content: m.content.clone(),
        })
        .collect()
}

fn gen_options(
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    seed: Option<u64>,
    stream: bool,
) -> GenOptions {
    GenOptions {
        max_tokens: max_tokens.unwrap_or(128).clamp(1, 4096),
        temperature: temperature.unwrap_or(0.7),
        top_p: top_p.unwrap_or(0.9),
        seed: seed.unwrap_or(42),
        stream,
        ..Default::default()
    }
}

fn chat_completion<S: Write>(
    stream: &mut S,
    cors: Option<&str>,
    state: &mut ServerState,
    req: ChatRequest,
) -> Result<()> {
    let msgs = messages(&req);
    if msgs.is_empty() {
        return json_error(
            stream,
            cors,
            "400 Bad Request",
            "messages no puede estar vacío",
        );
    }
    let stream_mode = req.stream.unwrap_or(false);
    let opts = gen_options(
        req.max_tokens,
        req.temperature,
        req.top_p,
        req.seed,
        stream_mode,
    );
    let id = format!("chatcmpl-{}", now_secs());
    let model = req
        .model
        .clone()
        .unwrap_or_else(|| state.model_name.clone());
    if stream_mode {
        write_headers(
            stream,
            cors,
            "200 OK",
            "text/event-stream",
            &[("Cache-Control", "no-cache")],
        )?;
        let first = json!({
            "id": id, "object": "chat.completion.chunk", "created": now_secs(), "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
        });
        write!(stream, "data: {}\n\n", serde_json::to_string(&first)?)?;
        let mut error = None;
        let result = state.session.respond_chat(&msgs, &opts, |piece| {
            let chunk = json!({
                "id": id, "object": "chat.completion.chunk", "created": now_secs(), "model": model,
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": null}]
            });
            if let Ok(text) = serde_json::to_string(&chunk) {
                if write!(stream, "data: {text}\n\n").is_err() || stream.flush().is_err() {
                    error = Some(Error::new("el cliente cerró el stream"));
                }
            }
        });
        match result {
            Ok((_, metrics)) => {
                let last = json!({
                    "id": id, "object": "chat.completion.chunk", "created": now_secs(), "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": metrics.prompt_tokens, "completion_tokens": metrics.generated_tokens, "total_tokens": metrics.prompt_tokens + metrics.generated_tokens}
                });
                write!(stream, "data: {}\n\n", serde_json::to_string(&last)?)?;
                write!(stream, "data: [DONE]\n\n")?;
                stream.flush()?;
                Ok(())
            }
            Err(e) => {
                if let Some(err) = error {
                    return Err(err);
                }
                write!(
                    stream,
                    "data: {}\n\n",
                    serde_json::to_string(&json!({"error": {"message": e.to_string()}}))?
                )?;
                Ok(())
            }
        }
    } else {
        match state.session.respond_chat(&msgs, &opts, |_| {}) {
            Ok((content, metrics)) => write_json(
                stream,
                cors,
                "200 OK",
                &json!({
                    "id": id, "object": "chat.completion", "created": now_secs(), "model": model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": metrics.prompt_tokens, "completion_tokens": metrics.generated_tokens, "total_tokens": metrics.prompt_tokens + metrics.generated_tokens}
                }),
            ),
            Err(e) => json_error(stream, cors, "500 Internal Server Error", &e.to_string()),
        }
    }
}

fn text_completion<S: Write>(
    stream: &mut S,
    cors: Option<&str>,
    state: &mut ServerState,
    req: CompletionRequest,
) -> Result<()> {
    let model = req
        .model
        .clone()
        .unwrap_or_else(|| state.model_name.clone());
    let msgs = vec![crate::tokenizer::ChatMessage {
        role: crate::tokenizer::Role::User,
        content: req.prompt,
    }];
    let stream_mode = req.stream.unwrap_or(false);
    let opts = gen_options(
        req.max_tokens,
        req.temperature,
        req.top_p,
        req.seed,
        stream_mode,
    );
    let id = format!("cmpl-{}", now_secs());
    if stream_mode {
        write_headers(
            stream,
            cors,
            "200 OK",
            "text/event-stream",
            &[("Cache-Control", "no-cache")],
        )?;
        let result = state.session.respond_chat(&msgs, &opts, |piece| {
            let chunk = json!({
                "id": id, "object": "text_completion", "created": now_secs(), "model": model,
                "choices": [{"index": 0, "text": piece, "finish_reason": null}]
            });
            if let Ok(text) = serde_json::to_string(&chunk) {
                let _ = write!(stream, "data: {text}\n\n");
                let _ = stream.flush();
            }
        });
        match result {
            Ok(_) => {
                let last = json!({
                    "id": id, "object": "text_completion", "created": now_secs(), "model": model,
                    "choices": [{"index": 0, "text": "", "finish_reason": "stop"}]
                });
                write!(stream, "data: {}\n\n", serde_json::to_string(&last)?)?;
                write!(stream, "data: [DONE]\n\n")?;
                stream.flush()?;
                Ok(())
            }
            Err(e) => json_error(stream, cors, "500 Internal Server Error", &e.to_string()),
        }
    } else {
        match state.session.respond_chat(&msgs, &opts, |_| {}) {
            Ok((text, metrics)) => write_json(
                stream,
                cors,
                "200 OK",
                &json!({
                    "id": id, "object": "text_completion", "created": now_secs(), "model": model,
                    "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": metrics.prompt_tokens, "completion_tokens": metrics.generated_tokens, "total_tokens": metrics.prompt_tokens + metrics.generated_tokens}
                }),
            ),
            Err(e) => json_error(stream, cors, "500 Internal Server Error", &e.to_string()),
        }
    }
}

fn status_json(state: &ServerState) -> Value {
    let cfg = state.session.model.config();
    json!({
        "model": state.model_name,
        "density": state.session.model.density(),
        "payload_bytes": state.session.model.payload_len(),
        "hidden": cfg.hidden, "layers": cfg.layers, "heads": cfg.heads,
        "kv_heads": cfg.kv_heads, "experts": cfg.experts, "top_k": cfg.top_k,
        "ctx": state.session.cache.capacity(),
        "rss_mb": rss_bytes() as f64 / (1024.0 * 1024.0),
        "uptime_s": state.started.elapsed().as_secs(),
    })
}

fn worlds_json(state: &ServerState) -> Value {
    match &state.taxonomy {
        None => json!({"worlds": []}),
        Some(tax) => json!({
            "feature_dim": tax.feature_dim,
            "worlds": tax.worlds.iter().map(|w| json!({
                "id": w.id, "label": w.label, "priority": w.priority,
                "provisional": w.provisional,
                "examples": w.examples.len(),
                "experts": w.experts,
                "experts_by_layer": w.experts_by_layer,
                "routes": w.routes,
            })).collect::<Vec<_>>()
        }),
    }
}

fn deploy_world(state: &mut ServerState, req: DeployRequest) -> Result<Value> {
    let tax = state
        .taxonomy
        .as_ref()
        .ok_or_else(|| Error::new("no hay taxonomía de mundos cargada"))?;
    let cfg = state.session.model.config().clone();
    let mask = expert_mask_for(tax, &req.worlds, cfg.layers, cfg.experts)?;
    let worlds: Vec<ActivatedWorld> = tax
        .worlds
        .iter()
        .filter(|w| req.worlds.iter().any(|id| id == &w.id))
        .map(|w| ActivatedWorld {
            id: w.id.clone(),
            label: w.label.clone().unwrap_or_else(|| w.id.clone()),
            priority: w.priority,
            provisional: w.provisional,
            probability: 1.0,
        })
        .collect();
    let activation = Activation {
        worlds,
        incomplete: false,
        warning: None,
        probabilities: Vec::new(),
    };
    let plan = build_plan(
        tax,
        &activation,
        &state.session.model.ukj.header.routes,
        state.session.model.tensor_infos(),
        cfg.experts,
        state.session.model.payload_len(),
    )?;
    let stats = state.session.model.load_plan(&plan);
    state.session.model.set_expert_mask(Some(mask.clone()))?;
    let experts_per_layer = mask
        .iter()
        .map(|bits| (0..cfg.experts).filter(|e| bits & (1u64 << e) != 0).count())
        .max()
        .unwrap_or(0);
    Ok(json!({
        "ok": true,
        "worlds": req.worlds,
        "experts_per_layer": experts_per_layer,
        "loaded_bytes": plan.loaded_bytes,
        "ranges": plan.ranges.len(),
        "rss_before": stats.rss_before,
        "rss_after": stats.rss_after,
        "rss_delta": stats.rss_delta,
        "elapsed_ms": stats.elapsed_ms,
        "mask": mask,
    }))
}

fn retrain_json(state: &ServerState) -> Result<Value> {
    let tax = state
        .taxonomy
        .as_ref()
        .ok_or_else(|| Error::new("no hay taxonomía de mundos cargada"))?;
    let router = TinyRouter::train(tax)?;
    let report =
        crate::router::evaluate_holdout(tax, 4, &crate::router::ActivationPolicy::default())?;
    Ok(json!({
        "ok": true,
        "worlds": router.world_ids.len(),
        "accuracy": report.accuracy,
        "train_examples": report.train_examples,
        "test_examples": report.test_examples,
    }))
}

fn handle<S: Read + Write>(stream: &mut S, state: &mut ServerState) -> Result<()> {
    let Some(req) = read_request(stream)? else {
        return Ok(());
    };
    let cors = cors_origin(&req, state.tls);
    let cors = cors.as_deref();
    if req.method == "OPTIONS" {
        write_headers(stream, cors, "204 No Content", "text/plain", &[])?;
        return Ok(());
    }
    if needs_auth(&req.path) && !authorized(&req, state) {
        return json_error(stream, cors, "401 Unauthorized", "API key inválida");
    }
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/") | ("GET", "/index.html") => {
            let html = index_html(state);
            write_headers(
                stream,
                cors,
                "200 OK",
                "text/html; charset=utf-8",
                &[("Content-Length", &html.len().to_string())],
            )?;
            stream.write_all(html.as_bytes())?;
            stream.flush()?;
            Ok(())
        }
        ("GET", "/health") => write_json(stream, cors, "200 OK", &json!({"ok": true})),
        ("GET", "/api/status") => write_json(stream, cors, "200 OK", &status_json(state)),
        ("GET", "/api/worlds") => write_json(stream, cors, "200 OK", &worlds_json(state)),
        ("POST", "/api/worlds/deploy") => {
            let body: DeployRequest = serde_json::from_slice(&req.body)
                .map_err(|e| Error::new(format!("JSON inválido: {e}")))?;
            match deploy_world(state, body) {
                Ok(value) => write_json(stream, cors, "200 OK", &value),
                Err(e) => json_error(stream, cors, "400 Bad Request", &e.to_string()),
            }
        }
        ("POST", "/api/worlds/retrain") => match retrain_json(state) {
            Ok(value) => write_json(stream, cors, "200 OK", &value),
            Err(e) => json_error(stream, cors, "400 Bad Request", &e.to_string()),
        },
        ("GET", "/v1/models") => write_json(
            stream,
            cors,
            "200 OK",
            &json!({
                "object": "list",
                "data": [{"id": state.model_name, "object": "model", "owned_by": "cabrita", "created": now_secs()}]
            }),
        ),
        ("POST", "/v1/chat/completions") => {
            let body: ChatRequest = serde_json::from_slice(&req.body)
                .map_err(|e| Error::new(format!("JSON inválido: {e}")))?;
            chat_completion(stream, cors, state, body)
        }
        ("POST", "/v1/completions") => {
            let body: CompletionRequest = serde_json::from_slice(&req.body)
                .map_err(|e| Error::new(format!("JSON inválido: {e}")))?;
            text_completion(stream, cors, state, body)
        }
        _ => json_error(stream, cors, "404 Not Found", "ruta no encontrada"),
    }
}

fn tls_config(cert: &Path, key: &Path) -> Result<Arc<rustls::ServerConfig>> {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let certs = rustls_pemfile::certs(&mut BufReader::new(File::open(cert).map_err(|e| {
        Error::new(format!(
            "no se puede leer el certificado {}: {e}",
            cert.display()
        ))
    })?))
    .collect::<std::result::Result<Vec<_>, _>>()
    .map_err(|e| Error::new(format!("certificado TLS inválido: {e}")))?;
    let key =
        rustls_pemfile::private_key(&mut BufReader::new(File::open(key).map_err(|e| {
            Error::new(format!("no se puede leer la clave {}: {e}", key.display()))
        })?))
        .map_err(|e| Error::new(format!("clave TLS inválida: {e}")))?
        .ok_or_else(|| Error::new("la clave TLS no contiene una clave privada"))?;
    let config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map_err(|e| Error::new(format!("configuración TLS inválida: {e}")))?;
    Ok(Arc::new(config))
}

pub struct ServeOptions {
    pub model: PathBuf,
    pub addr: String,
    pub ctx: usize,
    pub worlds: Option<PathBuf>,
    pub api_key: Option<String>,
    pub verify_payload: bool,
    pub tls_cert: Option<PathBuf>,
    pub tls_key: Option<PathBuf>,
}

pub fn serve(opts: ServeOptions) -> Result<()> {
    let model = crate::model::OlmoeModel::open_with_verify(
        &opts.model,
        opts.verify_payload,
        opts.verify_payload,
    )?;
    let model_name = model.ukj.model_id().to_string();
    let session = ChatSession::new(model, opts.ctx, None)?;
    let taxonomy = find_taxonomy(opts.worlds.as_deref());
    let tls = match (opts.tls_cert.as_deref(), opts.tls_key.as_deref()) {
        (Some(cert), Some(key)) => Some(tls_config(cert, key)?),
        (None, None) => None,
        _ => return Err(Error::new("para HTTPS pasa --tls-cert y --tls-key juntos")),
    };
    let state = ServerState {
        session,
        model_name,
        taxonomy,
        api_key: opts.api_key,
        tls: tls.is_some(),
        started: Instant::now(),
    };
    let listener = TcpListener::bind(&opts.addr)
        .map_err(|e| Error::new(format!("no se puede escuchar en {}: {e}", opts.addr)))?;
    let scheme = if tls.is_some() { "https" } else { "http" };
    eprintln!(
        "Cabrita: {} en {scheme}://{} · GUI y API OpenAI",
        state.model_name, opts.addr
    );
    let mut state = state;
    for incoming in listener.incoming() {
        match incoming {
            Ok(mut stream) => {
                // Un cliente ocioso o atascado se cierra en vez de bloquear la GUI y la API.
                let _ = stream.set_read_timeout(Some(IO_TIMEOUT));
                let _ = stream.set_write_timeout(Some(IO_TIMEOUT));
                if let Some(config) = &tls {
                    match rustls::ServerConnection::new(config.clone()) {
                        Ok(conn) => {
                            let mut tls_stream = rustls::StreamOwned::new(conn, stream);
                            if let Err(e) = handle(&mut tls_stream, &mut state) {
                                eprintln!("https: {e}");
                            }
                        }
                        Err(e) => eprintln!("https: TLS: {e}"),
                    }
                } else if let Err(e) = handle(&mut stream, &mut state) {
                    let _ = json_error(
                        &mut stream,
                        None,
                        "500 Internal Server Error",
                        &e.to_string(),
                    );
                    eprintln!("http: {e}");
                }
            }
            Err(e) => eprintln!("http: conexión fallida: {e}"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_is_embedded() {
        assert!(INDEX_HTML.contains("Cabrita"));
        assert!(INDEX_HTML.contains("/api/worlds/deploy"));
    }

    #[test]
    fn find_taxonomy_none_is_none() {
        assert!(find_taxonomy(Some(Path::new("/no/existe.json"))).is_none());
    }

    #[test]
    fn auth_covers_v1_and_api_routes() {
        assert!(needs_auth("/v1/models"));
        assert!(needs_auth("/v1/chat/completions"));
        assert!(needs_auth("/api/status"));
        assert!(needs_auth("/api/worlds/deploy"));
        assert!(!needs_auth("/"));
        assert!(!needs_auth("/health"));
    }

    #[test]
    fn cors_only_allows_the_server_origin() {
        let request = |origin: &str| Request {
            method: "GET".to_string(),
            path: "/api/status".to_string(),
            headers: HashMap::from([
                ("origin".to_string(), origin.to_string()),
                ("host".to_string(), "127.0.0.1:8080".to_string()),
            ]),
            body: Vec::new(),
        };
        assert_eq!(
            cors_origin(&request("http://127.0.0.1:8080"), false).as_deref(),
            Some("http://127.0.0.1:8080")
        );
        assert!(cors_origin(&request("http://malicioso.example"), false).is_none());
        // El mismo Host por HTTPS no coincide con un Origin http://.
        assert!(cors_origin(&request("http://127.0.0.1:8080"), true).is_none());
        // Cabecera manipulada con CRLF: nunca se refleja.
        assert!(cors_origin(&request("http://127.0.0.1:8080\r\nX: y"), false).is_none());
    }

    #[test]
    fn gui_escapes_world_fields() {
        assert!(INDEX_HTML.contains("const esc = (s) =>"));
        assert!(INDEX_HTML.contains("esc(world.label||world.id)"));
        assert!(INDEX_HTML.contains("esc(world.id)"));
    }
}
