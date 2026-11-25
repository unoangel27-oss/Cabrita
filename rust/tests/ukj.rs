use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use cabrita::router::{TinyRouter, WorldTaxonomy};
use cabrita::{
    ensure_subs, extract_subs, inspect_report, parse_density, verify_subs, SubsFile, UkjFile,
};
use sha2::{Digest, Sha256};
use tempfile::tempdir;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn copy_fixture(dir: &Path) -> PathBuf {
    let target = dir.join("tiny.ukj");
    fs::copy(fixture("tiny.ukj"), &target).unwrap();
    target
}

fn flip(path: &Path, offset: u64) {
    let mut f = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    f.seek(SeekFrom::Start(offset)).unwrap();
    let mut b = [0u8; 1];
    f.read_exact(&mut b).unwrap();
    f.seek(SeekFrom::Start(offset)).unwrap();
    f.write_all(&[b[0] ^ 0xFF]).unwrap();
}

fn rehash_ukj(path: &Path) {
    let ukj = UkjFile::open(path, false, false).unwrap();
    let mut f = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    let mut payload = vec![0u8; ukj.container.payload_len as usize];
    f.seek(SeekFrom::Start(ukj.container.payload_offset))
        .unwrap();
    f.read_exact(&mut payload).unwrap();
    let payload_sha = Sha256::digest(&payload);
    f.seek(SeekFrom::Start(56)).unwrap();
    f.write_all(&payload_sha).unwrap();

    let total = ukj.container.payload_offset + ukj.container.payload_len;
    f.seek(SeekFrom::Start(0)).unwrap();
    let mut h = Sha256::new();
    let mut buf = [0u8; 8192];
    let mut remaining = total;
    while remaining > 0 {
        let n = std::cmp::min(buf.len() as u64, remaining) as usize;
        f.read_exact(&mut buf[..n]).unwrap();
        h.update(&buf[..n]);
        remaining -= n as u64;
    }
    f.seek(SeekFrom::Start(total)).unwrap();
    f.write_all(&h.finalize()).unwrap();
}

#[test]
fn parse_and_inspect_fixture() {
    let ukj = UkjFile::open(&fixture("tiny.ukj"), true, true).unwrap();
    assert_eq!(ukj.header.format, "UKJ");
    assert_eq!(ukj.header.version, 1);
    assert_eq!(ukj.header.model.id, "tiny-p3_p");
    assert_eq!(ukj.header.tensors.len(), 3);
    assert_eq!(ukj.header.routes.len(), 2);
    assert_eq!(ukj.header.densities.default, "p3_p");
    assert_eq!(ukj.header.payload.length, ukj.container.payload_len);
    let nsubs: usize = ukj.header.tensors.iter().map(|t| t.subs.len()).sum();
    assert_eq!(nsubs, 8);
    for t in &ukj.header.tensors {
        assert_eq!(t.offset % 64, 0, "{} no está alineado a 64", t.name);
        assert_eq!(
            t.subs.iter().map(|s| s.nbytes).sum::<u64>(),
            t.nbytes,
            "{}",
            t.name
        );
    }
    let report = inspect_report(&ukj, true, None);
    assert_eq!(report["tensor_count"], 3);
    assert_eq!(report["subparameter_count"], 8);
    assert_eq!(report["checks"]["payload_sha256"], true);
    let encoded = serde_json::to_vec(&report).unwrap();
    let decoded: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded["model_id"], "tiny-p3_p");
}

#[test]
fn density_codes() {
    assert_eq!(parse_density("p_p").unwrap(), 1);
    assert_eq!(parse_density("P5_P").unwrap(), 5);
    assert_eq!(parse_density("p9_p").unwrap(), 9);
    assert_eq!(parse_density("p12_p").unwrap(), 12);
    assert!(parse_density("p0_p").is_err());
    assert!(parse_density("p13_p").is_err());
    assert!(parse_density("q4_k").is_err());
}

#[test]
fn subs_extract_verify_and_repair() {
    let dir = tempdir().unwrap();
    let ukj = fixture("tiny.ukj");
    let subs = dir.path().join("subs.subs");

    extract_subs(&ukj, Some(&subs), None, true).unwrap();
    let status = verify_subs(&ukj, Some(&subs), None).unwrap();
    assert!(status.ok, "{}", status.message);
    assert_eq!(status.entries, 8);

    // El .subs generado por Python también debe validar en Rust.
    let python_subs = fixture("tiny.subs");
    let status = verify_subs(&ukj, Some(&python_subs), None).unwrap();
    assert!(status.ok, "{}", status.message);

    fs::remove_file(&subs).unwrap();
    let status = ensure_subs(&ukj, Some(&subs), None).unwrap();
    assert!(status.ok, "{}", status.message);
    assert_eq!(status.action.as_deref(), Some("creado"));

    let sf = SubsFile::open(&subs, false, false).unwrap();
    let first = &sf.header.entries[0];
    flip(&subs, sf.container.payload_offset + first.offset);
    let status = verify_subs(&ukj, Some(&subs), None).unwrap();
    assert!(!status.ok);
    assert_eq!(status.status, "invalid");

    let status = ensure_subs(&ukj, Some(&subs), None).unwrap();
    assert!(status.ok, "{}", status.message);
    assert_eq!(status.action.as_deref(), Some("reparado"));
    assert!(verify_subs(&ukj, Some(&subs), None).unwrap().ok);
}

#[test]
fn desynced_subs_is_repaired() {
    let dir = tempdir().unwrap();
    let ukj = copy_fixture(dir.path());
    let subs = dir.path().join("subs.subs");
    extract_subs(&ukj, Some(&subs), None, true).unwrap();
    assert!(verify_subs(&ukj, Some(&subs), None).unwrap().ok);

    let parsed = UkjFile::open(&ukj, false, false).unwrap();
    flip(&ukj, parsed.container.payload_offset + 1);
    rehash_ukj(&ukj);

    let status = verify_subs(&ukj, Some(&subs), None).unwrap();
    assert!(!status.ok);
    assert_eq!(status.status, "desync");

    let status = ensure_subs(&ukj, Some(&subs), None).unwrap();
    assert!(status.ok, "{}", status.message);
    assert_eq!(status.action.as_deref(), Some("reparado"));
    assert!(verify_subs(&ukj, Some(&subs), None).unwrap().ok);
}

#[test]
fn invalid_files_are_rejected() {
    let dir = tempdir().unwrap();

    let truncated = copy_fixture(dir.path());
    let len = fs::metadata(&truncated).unwrap().len();
    fs::OpenOptions::new()
        .write(true)
        .open(&truncated)
        .unwrap()
        .set_len(len - 1)
        .unwrap();
    assert!(UkjFile::open(&truncated, false, true).is_err());

    let bad_header = dir.path().join("bad-header.ukj");
    fs::copy(fixture("tiny.ukj"), &bad_header).unwrap();
    flip(&bad_header, 88 + 5);
    assert!(UkjFile::open(&bad_header, false, true).is_err());

    let bad_payload = dir.path().join("bad-payload.ukj");
    fs::copy(fixture("tiny.ukj"), &bad_payload).unwrap();
    let ukj = UkjFile::open(&bad_payload, false, false).unwrap();
    flip(&bad_payload, ukj.container.payload_offset + 1);
    assert!(UkjFile::open(&bad_payload, true, true).is_err());

    let bad_magic = dir.path().join("bad-magic.ukj");
    fs::copy(fixture("tiny.ukj"), &bad_magic).unwrap();
    let mut f = fs::OpenOptions::new().write(true).open(&bad_magic).unwrap();
    f.seek(SeekFrom::Start(0)).unwrap();
    f.write_all(b"NOPE").unwrap();
    assert!(UkjFile::open(&bad_magic, false, true).is_err());
}

#[test]
fn cli_inspect_and_verify() {
    let out = Command::new(env!("CARGO_BIN_EXE_cabrita"))
        .args(["inspect", fixture("tiny.ukj").to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(value["tensor_count"], 3);
    assert_eq!(value["subparameter_count"], 8);

    let out = Command::new(env!("CARGO_BIN_EXE_cabrita"))
        .args(["verify", fixture("tiny.ukj").to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn cli_chat_routes_without_world_flag() {
    let out = Command::new(env!("CARGO_BIN_EXE_cabrita"))
        .args([
            "chat",
            fixture("tiny-olmoe-continue.ukj").to_str().unwrap(),
            "--prompt",
            "escribe una función en Python",
            "--worlds",
            fixture("tiny-worlds.json").to_str().unwrap(),
            "--max-tokens",
            "1",
            "--no-stream",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let router = &value["router"];
    assert_eq!(router["incomplete"], false);
    assert_eq!(router["worlds"][0]["id"], "codigo");
    assert_eq!(
        router["experts_per_layer"], 1,
        "sin --world debe enmascarar los expertos del mundo elegido"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("router:"), "{stderr}");
}

#[test]
fn cli_chat_unknown_text_uses_incomplete_mode() {
    let out = Command::new(env!("CARGO_BIN_EXE_cabrita"))
        .args([
            "chat",
            fixture("tiny-olmoe-continue.ukj").to_str().unwrap(),
            "--prompt",
            "qzxv wplk ñandú 88",
            "--worlds",
            fixture("tiny-worlds.json").to_str().unwrap(),
            "--max-tokens",
            "1",
            "--no-stream",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(value["router"]["incomplete"], true);
    assert_eq!(value["router"]["experts_per_layer"], 2);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("modo incompleto"), "{stderr}");
}

#[test]
fn cli_chat_rejects_corrupted_payload() {
    let dir = tempdir().unwrap();
    let bad = dir.path().join("bad-chat.ukj");
    fs::copy(fixture("tiny-olmoe-continue.ukj"), &bad).unwrap();
    let ukj = UkjFile::open(&bad, false, false).unwrap();
    flip(&bad, ukj.container.payload_offset + 1);

    let out = Command::new(env!("CARGO_BIN_EXE_cabrita"))
        .args([
            "chat",
            bad.to_str().unwrap(),
            "--prompt",
            "hola",
            "--max-tokens",
            "1",
            "--no-stream",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success(), "chat aceptó un payload corrupto");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("payload no coincide"),
        "se esperaba el error de checksum: {stderr}"
    );
}

#[test]
fn cli_verify_rejects_truncated_file() {
    let dir = tempdir().unwrap();
    let truncated = dir.path().join("truncated.ukj");
    fs::copy(fixture("tiny.ukj"), &truncated).unwrap();
    let len = fs::metadata(&truncated).unwrap().len();
    fs::OpenOptions::new()
        .write(true)
        .open(&truncated)
        .unwrap()
        .set_len(len - 1)
        .unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_cabrita"))
        .args(["verify", truncated.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("truncado") || stderr.contains("tamaño"),
        "mensaje poco claro: {stderr}"
    );
}

#[test]
fn cli_chat_missing_world_has_clear_message() {
    let out = Command::new(env!("CARGO_BIN_EXE_cabrita"))
        .args([
            "chat",
            fixture("tiny-olmoe-continue.ukj").to_str().unwrap(),
            "--prompt",
            "hola",
            "--world",
            "noexiste",
            "--worlds",
            fixture("tiny-worlds.json").to_str().unwrap(),
            "--max-tokens",
            "1",
            "--no-stream",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("mundo 'noexiste' no está en la taxonomía"),
        "mensaje poco claro: {stderr}"
    );
}

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    port
}

/// Envía una petición y devuelve la respuesta completa (estado, cabeceras y cuerpo).
fn http_with(
    addr: &str,
    method: &str,
    path: &str,
    body: Option<&str>,
    headers: &[(&str, &str)],
) -> String {
    let mut stream = TcpStream::connect(addr).unwrap();
    let body = body.unwrap_or("");
    let mut head = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n",
        body.len()
    );
    for (k, v) in headers {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes()).unwrap();
    stream.write_all(body.as_bytes()).unwrap();
    let mut raw = String::new();
    stream.read_to_string(&mut raw).unwrap();
    raw
}

/// Igual que [`http_with`] pero solo devuelve el cuerpo.
fn http(addr: &str, method: &str, path: &str, body: Option<&str>) -> String {
    http_with(addr, method, path, body, &[])
        .split("\r\n\r\n")
        .nth(1)
        .unwrap_or("")
        .to_string()
}

/// Espera a que el servidor conteste `/health`.
fn wait_for_server(addr: &str) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Ok(mut stream) = TcpStream::connect(addr) {
            let _ = write!(
                stream,
                "GET /health HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n"
            );
            let mut raw = String::new();
            let _ = stream.read_to_string(&mut raw);
            if raw.contains("200 OK") {
                return;
            }
        }
        assert!(Instant::now() < deadline, "el servidor no arrancó");
        thread::sleep(Duration::from_millis(50));
    }
}

struct ServerGuard(std::process::Child);
impl Drop for ServerGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn cli_serve_openai_and_gui() {
    let port = free_port();
    let addr = format!("127.0.0.1:{port}");
    let child = Command::new(env!("CARGO_BIN_EXE_cabrita"))
        .args([
            "serve",
            fixture("tiny-olmoe-continue.ukj").to_str().unwrap(),
            "--addr",
            &addr,
            "--ctx",
            "64",
            "--worlds",
            fixture("tiny-worlds.json").to_str().unwrap(),
            "--no-verify-payload",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let _guard = ServerGuard(child);

    wait_for_server(&addr);

    let status = http(&addr, "GET", "/api/status", None);
    assert!(status.contains("\"model\":\"tiny-olmoe-p8_p\""), "{status}");
    let models = http(&addr, "GET", "/v1/models", None);
    assert!(models.contains("tiny-olmoe-p8_p"), "{models}");
    let page = http(&addr, "GET", "/", None);
    assert!(page.contains("Cabrita"));
    assert!(page.contains("Deploy weights"));

    let body = "{\"model\":\"cabrita\",\"messages\":[{\"role\":\"user\",\"content\":\"hola\"}],\"max_tokens\":1}";
    let response = http(&addr, "POST", "/v1/chat/completions", Some(body));
    let value: serde_json::Value = serde_json::from_str(&response).unwrap();
    assert_eq!(value["object"], "chat.completion");
    assert_eq!(value["choices"][0]["message"]["role"], "assistant");

    let deployed = http(
        &addr,
        "POST",
        "/api/worlds/deploy",
        Some("{\"worlds\":[\"codigo\"]}"),
    );
    let value: serde_json::Value = serde_json::from_str(&deployed).unwrap();
    assert_eq!(value["ok"], true);
    assert_eq!(value["worlds"][0], "codigo");
}

#[test]
fn cli_serve_requires_api_key_and_limits_cors() {
    let port = free_port();
    let addr = format!("127.0.0.1:{port}");
    let child = Command::new(env!("CARGO_BIN_EXE_cabrita"))
        .args([
            "serve",
            fixture("tiny-olmoe-continue.ukj").to_str().unwrap(),
            "--addr",
            &addr,
            "--ctx",
            "64",
            "--worlds",
            fixture("tiny-worlds.json").to_str().unwrap(),
            "--api-key",
            "secreto-de-prueba",
            "--no-verify-payload",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let _guard = ServerGuard(child);
    wait_for_server(&addr);

    let key = ("Authorization", "Bearer secreto-de-prueba");
    // /v1/* sigue exigiendo clave.
    assert!(http_with(&addr, "GET", "/v1/models", None, &[]).contains("401"));
    assert!(http_with(&addr, "GET", "/v1/models", None, &[key]).contains("200 OK"));

    // /api/* ahora también: antes quedaba abierto.
    assert!(http_with(&addr, "GET", "/api/status", None, &[]).contains("401"));
    let status = http_with(&addr, "GET", "/api/status", None, &[key]);
    assert!(status.contains("200 OK"), "{status}");
    assert!(status.contains("tiny-olmoe-p8_p"), "{status}");
    assert!(http_with(&addr, "POST", "/api/worlds/retrain", Some("{}"), &[]).contains("401"));
    assert!(http_with(&addr, "POST", "/api/worlds/retrain", Some("{}"), &[key]).contains("200 OK"));

    // La GUI se sirve sin clave y lleva la clave inyectada para sus fetch.
    let page = http_with(&addr, "GET", "/", None, &[]);
    assert!(page.contains("200 OK"), "{page}");
    assert!(
        page.contains(r#"const API_KEY = "secreto-de-prueba";"#),
        "{page}"
    );

    // CORS solo para el origen propio.
    let own = format!("http://{addr}");
    let same = http_with(&addr, "GET", "/api/status", None, &[key, ("Origin", &own)]);
    assert!(
        same.contains(&format!("Access-Control-Allow-Origin: {own}")),
        "{same}"
    );
    let foreign = http_with(
        &addr,
        "GET",
        "/api/status",
        None,
        &[key, ("Origin", "http://malicioso.example")],
    );
    assert!(
        !foreign.contains("Access-Control-Allow-Origin"),
        "{foreign}"
    );
    let preflight = http_with(
        &addr,
        "OPTIONS",
        "/api/worlds/deploy",
        None,
        &[("Origin", "http://malicioso.example")],
    );
    assert!(
        !preflight.contains("Access-Control-Allow-Origin"),
        "{preflight}"
    );
}

#[test]
fn cli_eval_perplexity_and_qa() {
    let dir = tempdir().unwrap();
    let text = dir.path().join("eval.txt");
    fs::write(&text, "hola mundo\n").unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_cabrita"))
        .args([
            "eval",
            "perplexity",
            fixture("tiny-olmoe-continue.ukj").to_str().unwrap(),
            "--text",
            text.to_str().unwrap(),
            "--ctx",
            "64",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(value["perplexity"].as_f64().unwrap_or(0.0) > 0.0);
    assert_eq!(value["density"], "p8_p");
    assert!(value["tokens"].as_u64().unwrap_or(0) > 0);

    let qa = dir.path().join("qa.jsonl");
    fs::write(
        &qa,
        "{\"prompt\":\"hola\",\"choices\":[\"mundo\",\"adios\"],\"answer\":\"mundo\"}\n",
    )
    .unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_cabrita"))
        .args([
            "eval",
            "qa",
            fixture("tiny-olmoe-continue.ukj").to_str().unwrap(),
            "--data",
            qa.to_str().unwrap(),
            "--ctx",
            "64",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(value["total"], 1);
    assert_eq!(value["correct"], 1);
    assert_eq!(value["items"][0]["predicted"], "mundo");
}

#[test]
fn cli_world_backup_restore_and_retrain() {
    let dir = tempdir().unwrap();
    let worlds = dir.path().join("tiny-worlds.json");
    fs::copy(fixture("tiny-worlds.json"), &worlds).unwrap();
    let backups = dir.path().join("backups");

    let out = Command::new(env!("CARGO_BIN_EXE_cabrita"))
        .args([
            "world",
            "backup",
            "--worlds",
            worlds.to_str().unwrap(),
            "--out",
            backups.to_str().unwrap(),
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let info: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let backup = PathBuf::from(info["backup"].as_str().unwrap());
    let manifest = PathBuf::from(info["manifest"].as_str().unwrap());
    assert!(backup.is_file());
    assert!(manifest.is_file());
    let bytes = fs::read(&backup).unwrap();
    assert_eq!(
        info["sha256"].as_str().unwrap(),
        hex::encode(Sha256::digest(&bytes))
    );
    assert!(WorldTaxonomy::load(&backup).is_ok());

    fs::write(&worlds, b"{\"version\":1,\"worlds\":[]}").unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_cabrita"))
        .args([
            "world",
            "restore",
            "--worlds",
            worlds.to_str().unwrap(),
            "--backup",
            backup.to_str().unwrap(),
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let restored = WorldTaxonomy::load(&worlds).unwrap();
    assert_eq!(restored.worlds.len(), 2);
    let value: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let checkpoint = value["checkpoint"].as_str().map(PathBuf::from);
    assert!(
        checkpoint.as_ref().map(|p| p.is_file()).unwrap_or(false),
        "la restauración debe dejar un checkpoint"
    );

    let router_path = dir.path().join("tiny.router.json");
    let out = Command::new(env!("CARGO_BIN_EXE_cabrita"))
        .args([
            "world",
            "retrain",
            "--worlds",
            worlds.to_str().unwrap(),
            "--out",
            router_path.to_str().unwrap(),
            "--eval",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let info: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(info["accuracy"].as_f64().unwrap_or(0.0) > 0.0);
    let raw = fs::read(&router_path).unwrap();
    let router: TinyRouter = serde_json::from_slice(&raw).unwrap();
    assert_eq!(router.world_ids.len(), 2);
}

/// Escribe un manifiesto válido para un backup, con el SHA-256 de sus bytes.
fn write_manifest(backup: &std::path::Path, bytes: &[u8]) {
    let manifest = backup.with_file_name(format!(
        "{}.manifest.json",
        backup.file_name().unwrap().to_string_lossy()
    ));
    let sha = hex::encode(Sha256::digest(bytes));
    fs::write(
        manifest,
        format!("{{\"sha256\":\"{sha}\",\"bytes\":{}}}", bytes.len()),
    )
    .unwrap();
}

#[test]
fn cli_world_restore_invalid_keeps_original() {
    let dir = tempdir().unwrap();
    let worlds = dir.path().join("tiny-worlds.json");
    fs::copy(fixture("tiny-worlds.json"), &worlds).unwrap();
    let original = fs::read(&worlds).unwrap();
    let bad_backup = dir.path().join("bad-backup.json");
    let bad_bytes = b"{\"version\":1,\"worlds\":[]}";
    fs::write(&bad_backup, bad_bytes).unwrap();
    // Con manifiesto correcto la restauración llega a la validación de taxonomía.
    write_manifest(&bad_backup, bad_bytes);

    let out = Command::new(env!("CARGO_BIN_EXE_cabrita"))
        .args([
            "world",
            "restore",
            "--worlds",
            worlds.to_str().unwrap(),
            "--backup",
            bad_backup.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert_eq!(fs::read(&worlds).unwrap(), original);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("no es una taxonomía válida"), "{stderr}");
}

#[test]
fn cli_world_restore_requires_manifest() {
    let dir = tempdir().unwrap();
    let worlds = dir.path().join("tiny-worlds.json");
    fs::copy(fixture("tiny-worlds.json"), &worlds).unwrap();
    let original = fs::read(&worlds).unwrap();
    // Backup con taxonomía válida pero sin manifiesto: ya no se acepta.
    let backup = dir.path().join("orphan-backup.json");
    fs::copy(fixture("tiny-worlds.json"), &backup).unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_cabrita"))
        .args([
            "world",
            "restore",
            "--worlds",
            worlds.to_str().unwrap(),
            "--backup",
            backup.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert_eq!(fs::read(&worlds).unwrap(), original);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("no tiene manifiesto"), "{stderr}");
}

#[test]
fn cli_subs_repair_leaves_a_checkpoint() {
    let dir = tempdir().unwrap();
    let ukj = copy_fixture(dir.path());
    let subs = dir.path().join("subs.subs");
    let out = Command::new(env!("CARGO_BIN_EXE_cabrita"))
        .args([
            "subs",
            "extract",
            ukj.to_str().unwrap(),
            "--out",
            subs.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let sf = SubsFile::open(&subs, false, false).unwrap();
    flip(
        &subs,
        sf.container.payload_offset + sf.header.entries[0].offset,
    );
    let out = Command::new(env!("CARGO_BIN_EXE_cabrita"))
        .args([
            "subs",
            "repair",
            ukj.to_str().unwrap(),
            "--path",
            subs.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let checkpoints = dir.path().join("checkpoints");
    let backups = fs::read_dir(&checkpoints)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("bak"))
        .count();
    assert!(
        backups >= 1,
        "no se creó ningún checkpoint en {checkpoints:?}"
    );
}
