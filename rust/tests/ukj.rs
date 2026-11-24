use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

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
    assert!(parse_density("p0_p").is_err());
    assert!(parse_density("p9_p").is_err());
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
